# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fsdet.layers.batch_norm import get_norm
from fsdet.structures.boxes import Boxes, pairwise_iou
from fsdet.data.catalog import MetadataCatalog
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.activation import ReLU

from fsdet.layers import ShapeSpec
from fsdet.utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn_outputs import RPNOutputs, find_top_rpn_proposals
from fsdet.utils.events import get_event_storage

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
"""
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.
"""


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)

@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            in_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        features_per_level = []

        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
            features_per_level.append(t)
        return pred_objectness_logits, pred_anchor_deltas, features_per_level

@RPN_HEAD_REGISTRY.register()
class ContrastRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels * num_cell_anchors, kernel_size=3, stride=1, padding=1)
        self.per_level_bn = nn.ModuleList([
            get_norm("BN", in_channels) for _ in input_shape
        ])
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            in_channels, box_dim, kernel_size=1, stride=1
        )

        self.num_cell_anchors = num_cell_anchors
        self.box_dim = box_dim
        self.strides = [7, 6, 5, 4, 3]

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        features_per_level = []
        for i, x in enumerate(features):
            B, C, H, W = x.shape
            t = F.relu(self.per_level_bn[i](self.conv(x).view(B, self.num_cell_anchors, C, H, W).flatten(0,1)))
            features_per_level.append(t)
            #t = t.view(B, self.num_cell_anchors, C//self.num_cell_anchors, H, W).flatten(0,1)
            pred_objectness_logits.append(self.objectness_logits(t).view(B, self.num_cell_anchors, H, W))
            pred_anchor_deltas.append(self.anchor_deltas(t).view(B, self.num_cell_anchors * self.box_dim, H, W))
        return pred_objectness_logits, pred_anchor_deltas, features_per_level

@RPN_HEAD_REGISTRY.register()
class WeightHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]
        self.num_cell_anchors = num_cell_anchors
        self.box_dim = box_dim
        self.strides = [7, 6, 5, 4, 3]
        self.register_buffer("fg_iou_cont_loss_avg", torch.zeros(5))
        self.register_buffer("bg_iou_cont_loss_avg", torch.ones(5))
        """
        self.weight_gen = nn.Sequential(
            *[nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)]
        )


        for l in self.weight_gen.modules():
            if isinstance(l, (nn.Conv2d)):
                nn.init.normal_(l.weight, std=0.01)
                nn.init.constant_(l.bias, 0)

        """

    def forward(self, feature, u_feature, anchors, outputs):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        gt_logit, gt_deltas = outputs._get_ground_truth()
        gt_logit = torch.stack(gt_logit)
        bg_sample = {}
        fg_sample = {}

        iou_contrast_loss = []

        st = 0
        for lvl, (f, u_f, a, ks) in enumerate(zip(feature, u_feature, anchors, self.strides)):
            B, C, H, W = f.shape
            u_f = nn.Unfold(kernel_size=(ks, ks), stride=ks)(u_f).view(B * self.num_cell_anchors, C // self.num_cell_anchors, ks, ks, -1)

            curr_gt_logit = gt_logit[:,st:st+H*W*self.num_cell_anchors]
            st+=H*W*self.num_cell_anchors
            curr_gt_logit = curr_gt_logit.view(-1,H, W, self.num_cell_anchors).permute(0, 3, 1, 2)
            u_a = nn.Unfold(kernel_size=(ks, ks), stride=ks)(a.tensor.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)).view(1, self.num_cell_anchors, self.box_dim, ks, ks, -1)
            u_gt = nn.Unfold(kernel_size=(ks, ks), stride=ks)(curr_gt_logit.float()).view(B, self.num_cell_anchors, ks, ks, -1)
            u_gt = u_gt.permute(0, 4, 1, 2, 3).reshape(-1, self.num_cell_anchors * ks * ks)

            sample_box = Boxes(u_a[0,...,1].permute(0, 2, 3, 1).flatten(0, -2))
            gt_iou_label = pairwise_iou(sample_box, sample_box)
            buf = u_f.view(B, self.num_cell_anchors, C, ks, ks, -1).permute(0, 5, 1, 3, 4, 2).reshape(-1, self.num_cell_anchors * ks * ks, C)
            buf_sim = torch.bmm(buf, buf.permute(0, 2, 1))
            normed_buf = (buf_sim - buf_sim.mean(dim=-1).unsqueeze(-1)) / buf_sim.std(dim=-1).unsqueeze(-1)

            fg_mask = (u_gt == 1).any(dim=1)
            bg_mask = (u_gt == 0).all(dim=1)
            if fg_mask.sum() > 0:
                fg_buf = normed_buf[fg_mask]
                fg_sample[lvl] = ((fg_buf/0.7).sigmoid().log() * gt_iou_label).flatten(1,2).mean(dim=1)

            if bg_mask.sum() > 0:
                sampled_bg_idx = torch.randint(high=(bg_mask).sum(), size=(10,))
                #bg_buf = normed_buf[bg_mask][sampled_bg_idx]
                bg_buf = normed_buf[bg_mask]
                bg_sample[lvl] = ((bg_buf/0.7).sigmoid().log() * gt_iou_label).flatten(1,2).mean(dim=1)
            
            #iou_contrast_loss.append(-((normed_buf / 0.7).sigmoid().log() * gt_iou_label.unsqueeze(0)).mean())

        fg_iou_avg = torch.zeros_like(self.fg_iou_cont_loss_avg)
        bg_iou_avg = torch.zeros_like(self.bg_iou_cont_loss_avg)
        fg_mask = torch.zeros_like(fg_iou_avg).bool()
        bg_mask = torch.zeros_like(bg_iou_avg).bool()

        #loss_mask = torch.zeros_like(iou_contrast_loss).bool()
        #iou_contrast_loss = torch.stack(iou_contrast_loss) 
        storage = get_event_storage()
        for lvl, _ in enumerate(self.strides):
            if lvl in bg_sample:
                storage.put_histogram_wi_term("bg_kl_{}".format(lvl), bg_sample[lvl])
                bg_iou_avg[lvl] = -bg_sample[lvl].mean()
                bg_mask[lvl] = True
            if lvl in fg_sample:
                storage.put_histogram_wi_term("fg_kl_{}".format(lvl), fg_sample[lvl])
                fg_iou_avg[lvl] = -fg_sample[lvl].mean()
                fg_mask[lvl] = True
        
        self.fg_iou_cont_loss_avg[fg_mask] = self.fg_iou_cont_loss_avg[fg_mask] * 0.9 + fg_iou_avg.detach()[fg_mask] * 0.1
        self.bg_iou_cont_loss_avg[bg_mask] = self.bg_iou_cont_loss_avg[bg_mask] * 0.9 + bg_iou_avg.detach()[bg_mask] * 0.1

        fg_iou_avg[~fg_mask] = self.fg_iou_cont_loss_avg[~fg_mask]
        bg_iou_avg[~bg_mask] = self.bg_iou_cont_loss_avg[~bg_mask]

        iou_cont_loss = torch.log(torch.exp((fg_iou_avg - bg_iou_avg) / self.bg_iou_cont_loss_avg + 0.5) + 1).mean()

        loss = {
            #"iou_contrast_loss": (iou_contrast_loss[loss_mask] / self.iou_cont_loss_avg[loss_mask]).mean()
            #"iou_contrast_loss": (iou_contrast_loss).mean()
            "iou_contrast_loss": iou_cont_loss
        }

        return loss
            
@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len        = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features             = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh              = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta          = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight             = cfg.MODEL.RPN.LOSS_WEIGHT

        self.cl_head_only            = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])
        self.meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        if len(self.meta.thing_classes) > len(self.meta.base_classes):
            novel_cls_list = []
            for i, cls_name in enumerate(self.meta.thing_classes):
                if cls_name in self.meta.novel_classes:
                    novel_cls_list.append(i)
            self.novel_mask = torch.zeros(len(self.meta.thing_classes)+1)
            self.novel_mask[novel_cls_list] = 1
            self.base_mask = 1 - self.novel_mask
            self.base_mask[-1] = 0
        else:
            self.novel_mask = None
            self.base_mask = None

        if cfg.MODEL.RPN.IOU_CONT:
            self.weight_gen_head = RPN_HEAD_REGISTRY.get("WeightHead")(cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances] or None
            loss: dict[Tensor]
        """
        log = {}
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        gt_class = [x.gt_classes for x in gt_instances] if gt_instances is not None else None

        del gt_instances
        features = [features[f] for f in self.in_features]
        # pred_objectness_logits: list of L tensor of shape [N, A, Hi, Wi]
        # pred_anchor_deltas: list of L tensor of shape [N, A*B, Hi, Wi]
        pred_objectness_logits, pred_anchor_deltas, feature_per_level = self.rpn_head(features)
        anchors = self.anchor_generator(features)

        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
            gt_class,
            [self.novel_mask, self.base_mask]
        )


        if self.training and not self.cl_head_only:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
            if hasattr(self, 'weight_gen_head'):
                iou_cont_loss = self.weight_gen_head(features, feature_per_level, anchors[0], outputs)
                losses.update(iou_cont_loss)
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),  # transform anchors to proposals by applying delta
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )
            # For RPN-only models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # For end-to-end models, the RPN proposals are an intermediate state
            # and this sorting is actually not needed. But the cost is negligible.
            # 但是要注意，end-to-end models 在后面进入 RoI 的 proposals 实际上会在 label_and_sample_proposals 再次被打乱，
            # 所以再以后用到的其实并不是按照 objectness 倒序排列的。
            inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
            proposals = [p[ind] for p, ind in zip(proposals, inds)]

        return proposals, losses, log
