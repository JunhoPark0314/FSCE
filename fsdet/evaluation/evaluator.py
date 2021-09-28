# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
from fsdet.modeling.postprocessing import detector_postprocess

from detectron2.structures.boxes import pairwise_iou
from fsdet.data.detection_utils import read_image
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch
from torchvision import models as tv_models
from torchvision.utils import save_image

from fsdet.utils.comm import is_main_process


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass

class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results
    
def correct_outputs(inputs, outputs, model):
    classifier = tv_models.resnet18(pretrained=True)
    classifier = classifier.to(model.device)

    from torchvision import datasets, transforms as T
    transform = T.Compose([T.Resize(224)])
    only_crop = T.Compose([T.Resize(224)])

    for img_info, inst in zip(inputs, outputs):
        path = img_info["file_name"]
        img = read_image(path, format="BGR")
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        croped_img = []
        corrected_result = []
        pred_boxes = inst["proposals"].proposal_boxes.tensor
        inst["proposals"].proposal_boxes.tensor = inst["proposals"].proposal_boxes.tensor.cpu()
        
        img_info["instances"].pred_boxes = img_info["instances"].gt_boxes
        new_img_info = detector_postprocess(img_info["instances"], img.shape[1], img.shape[2])
        gt_iou = pairwise_iou(new_img_info.pred_boxes, inst["proposals"].proposal_boxes)
        mask = gt_iou > 0.5

        for oid, pid in mask.nonzero():
            xmin, ymin, xmax, ymax = pred_boxes[pid].long()
            gt_cls = img_info["instances"].gt_classes[oid]
            # The inverse of data loading logic in `datasets/pascal_voc.py`
            xmin += 1
            ymin += 1

            cropped = transform(model.normalizer(img.cuda()[:, ymin:ymax, -xmax:-xmin])).unsqueeze(0)
            result = classifier(cropped)
            test = only_crop(img[:, ymin:ymax, -xmax:-xmin])
            save_image(img/256, "origin.png")
            save_image(test/256, "test.png")
            
            # Change class result of prediction
            # Check corrected accuracy according to IoU

def inference_on_dataset(model, data_loader, evaluator, prop_evaluator, use_cls=False):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()
    prop_evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs, proposals = model(inputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time

            if use_cls:
                correct_outputs(inputs, proposals, model)

            evaluator.process(inputs, outputs)
            prop_evaluator.process(inputs, proposals)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )
                #break

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    prop_results = prop_evaluator.evaluate()
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}

    if prop_results is None:
        prop_results = {}

    return results, prop_results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
