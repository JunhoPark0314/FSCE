# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from detectron2.structures.boxes import pairwise_iou
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from fsdet.data import MetadataCatalog
from fsdet.utils import comm
from fsdet.utils.logger import create_small_table
from detectron2.structures import Boxes

from .evaluator import DatasetEvaluator


class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        # add this two terms for calculating the mAP of different subset
        try:
            self._base_classes = meta.base_classes
            self._novel_classes = meta.novel_classes
        except AttributeError:
            self._base_classes = meta.thing_classes
            self._novel_classes = None
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        if "instances" in outputs[0]:
            self.process_instance(inputs, outputs)
        else:
            self.process_proposal(inputs, outputs)
    
    def process_instance(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )
                self._predictions["proposal"].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def process_proposal(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["proposals"].to(self._cpu_device)
            boxes = instances.proposal_boxes.tensor.numpy()
            scores = instances.objectness_logits.sigmoid().tolist()
            for box, score in zip(boxes, scores):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions["proposal"].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )
    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        if len(predictions.keys()) == 1:
            return self.evaluate_proposal(predictions, ap_limit=[100])
        else:
            ar_res = self.evaluate_proposal(predictions,)
            return self.evaluate_instance(predictions,)

    def evaluate_instance(self, predictions,):
        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_instance_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            aps_base = defaultdict(list)
            aps_novel = defaultdict(list)
            exist_base, exist_novel = False, False
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

                    if self._base_classes is not None and cls_name in self._base_classes:
                        aps_base[thresh].append(ap * 100)
                        exist_base = True

                    if self._novel_classes is not None and cls_name in self._novel_classes:
                        aps_novel[thresh].append(ap * 100)
                        exist_novel = True

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}

        # adding evaluation of the base and novel classes
        if exist_base:
            mAP_base = {iou: np.mean(x) for iou, x in aps_base.items()}
            ret["bbox"].update(
                {"bAP": np.mean(list(mAP_base.values())), "bAP50": mAP_base[50],
                 "bAP75": mAP_base[75]}
            )

        if exist_novel:
            mAP_novel = {iou: np.mean(x) for iou, x in aps_novel.items()}
            ret["bbox"].update({
                "nAP": np.mean(list(mAP_novel.values())), "nAP50": mAP_novel[50],
                "nAP75": mAP_novel[75]
            })

        # write per class AP to logger
        per_class_res = {self._class_names[idx]: ap for idx, ap in enumerate(aps[50])}

        self._logger.info("Evaluate per-class mAP50:\n"+create_small_table(per_class_res))
        self._logger.info("Evaluate overall bbox:\n"+create_small_table(ret["bbox"]))
        return ret

    def evaluate_proposal(self, predictions, ap_limit=None):
        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_proposal_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            ars = OrderedDict()
            aps = OrderedDict()  # iou -> ap per class

            lines = predictions.get("proposal", [""])
            if ap_limit == None:
                ap_limit = [None]

            with open(res_file_template.format("proposal"), "w") as f:
                f.write("\n".join(lines))

            class_split = {"all": self._class_names}
            if self._novel_classes is not None and len(self._class_names) == (len(self._base_classes) + len(self._novel_classes)):
                new_split = {"base":self._base_classes,
                             "novel":self._novel_classes}
                class_split.update(new_split)

            for split_id, cls_split in class_split.items():
                for limit in [100, 1000]:
                    ar = voc_eval_ar(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        limit,
                        cls_split
                    )
                    ars["{}_AR@{}".format(split_id,limit)] = ar * 100

                for ap_num in ap_limit:
                    aps[split_id+str(ap_num)] = defaultdict(list)
                    for thresh in range(50, 100, 5):
                        _, _, ap = voc_eval(
                            res_file_template,
                            self._anno_file_template,
                            self._image_set_path,
                            'proposal',
                            ovthresh=thresh / 100.0,
                            use_07_metric=self._is_2007,
                            cls_split=cls_split,
                            limit=ap_num
                        )
                        aps[split_id+str(ap_num)][thresh].append(ap * 100)
                    
                """
                if self._base_classes is not None and cls_name in self._base_classes:
                    aps_base[thresh].append(ap * 100)
                    exist_base = True

                if self._novel_classes is not None and cls_name in self._novel_classes:
                    aps_novel[thresh].append(ap * 100)
                    exist_novel = True
                """

        for split_id, _ in aps.items():
            ret = OrderedDict()
            mAP = {iou: np.mean(x) for iou, x in aps[split_id].items()}
            ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
            self._logger.info("Evaluate AP overall bbox {}:\n".format(split_id)+create_small_table(ret["bbox"]))
        self._logger.info("Evaluate AR overall bbox:\n"+create_small_table(ars))
        return ars

##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename, return_size=False):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)
    height = float(tree.findall('size')[0].find("height").text)
    width = float(tree.findall('size')[0].find("width").text)

    if return_size:
        return objects, (height, width)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval_ar(detpath, annopath, imagesetfile, limit, cls_split):
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    image_size = {}
    for imagename in imagenames:
        recs[imagename], image_size[imagename] = parse_rec(annopath.format(imagename), return_size=True)

    npos = 0
    whole_recs = {}
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] in cls_split]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        whole_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det, "image_size": image_size[imagename]}

    # read dets
    detfile = detpath.format("proposal")
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    prediction_bb_dict = defaultdict(list)
    prediction_score_dict = defaultdict(list)
    for x in splitlines:
        image_id = x[0]
        score = x[1]
        bbox = [float(z) for z in x[2:]]
        prediction_bb_dict[image_id].append(bbox)
        prediction_score_dict[image_id].append(float(score))

    for img_id in list(prediction_bb_dict.keys()):
        sorted_score, sorted_idx = torch.tensor(prediction_score_dict[img_id]).sort(descending=True)
        sorted_idx = sorted_idx[:limit]
        sorted_score = sorted_score[:limit]
        mask = sorted_score > 0.05
        prediction_bb_dict[img_id] = torch.tensor(prediction_bb_dict[img_id])[sorted_idx][mask]
        prediction_score_dict[img_id] = sorted_score[mask]
    
    gt_iou = []
    num_pos = 0

    for img_id, prediction_bb in prediction_bb_dict.items():
        R = whole_recs[img_id]
        gt_box = Boxes(whole_recs[img_id]['bbox'])
        num_pos += len(gt_box)
        pred_box = Boxes(prediction_bb)
        iou = pairwise_iou(pred_box, gt_box)

        _gt_iou = torch.zeros(len(gt_box))
        for j in range(min(len(pred_box), len(gt_box))):
            max_iou, argmax_iou = iou.max(dim=0)
            gt_ovr, gt_ind = max_iou.max(dim=0)
            assert gt_ovr >= 0

            box_ind = argmax_iou[gt_ind]
            _gt_iou[j] = iou[box_ind, gt_ind]
            assert _gt_iou[j] == gt_ovr

            iou[box_ind, :] = -1
            iou[:, gt_ind] = -1
        
        gt_iou.append(_gt_iou)
    
    gt_iou = (
        torch.cat(gt_iou, dim=0) if len(gt_iou) else torch.zeros(0, dtype=torch.float32)
    )
    gt_iou, _ = torch.sort(gt_iou)
    step = 0.05
    thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)

    for i, t in enumerate(thresholds):
        recalls[i] = (gt_iou >= t).float().sum() / float(num_pos)
    
    ar = recalls.mean()
    return ar 

def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, cls_split=None, limit=None):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0

    if classname != 'proposal':
        class_check = lambda x: x == classname
    else:
        class_check = lambda x: True
    
    if cls_split is not None:
        split_check = lambda x: x in cls_split
    else:
        split_check = lambda x: True

    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if class_check(obj["name"]) and split_check(obj["name"])]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    if limit is not None:
        idx = torch.zeros(1000).bool()
        idx[:limit] = 1
        num_img = len(BB) // len(idx)
        idx = idx.repeat(num_img)

        BB = BB[idx]
        confidence = confidence[idx]
    
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
