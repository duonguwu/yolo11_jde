"""Loss functions for JDE model."""

import torch
import torch.nn as nn
from ..utils.ops import make_anchors, dist2bbox, xywh2xyxy


class TaskAlignedAssigner:
    """Task-aligned assigner for JDE model."""
    
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, use_tags=True):
        """Initialize TaskAlignedAssigner."""
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.use_tags = use_tags
    
    def __call__(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt, gt_tags):
        """Assign targets to predictions."""
        # Simplified implementation - would need full implementation for training
        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        
        # Return dummy values for inference-only module
        target_labels = torch.zeros_like(pred_scores[..., 0])
        target_bboxes = torch.zeros_like(pred_bboxes)
        target_scores = torch.zeros_like(pred_scores)
        fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool, device=pred_scores.device)
        target_gt_idx = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=pred_scores.device)
        target_tags = torch.zeros(batch_size, num_anchors, 1, dtype=torch.long, device=pred_scores.device)
        
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx, target_tags


class BboxLoss:
    """Bounding box loss for JDE model."""
    
    def __init__(self, reg_max=16):
        """Initialize BboxLoss."""
        self.reg_max = reg_max
    
    def __call__(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Calculate bbox loss."""
        # Simplified implementation - would need full implementation for training
        device = pred_dist.device
        bbox_loss = torch.tensor(0.0, device=device)
        dfl_loss = torch.tensor(0.0, device=device)
        return bbox_loss, dfl_loss
    
    def to(self, device):
        """Move to device."""
        return self


class MetricLearningLoss:
    """Metric learning loss for ReID embeddings."""
    
    def __init__(self):
        """Initialize MetricLearningLoss."""
        pass
    
    def __call__(self, pred_embeds, target_tags, confidences):
        """Calculate embedding loss."""
        # Simplified implementation - would need full implementation for training
        device = pred_embeds.device
        embed_loss = torch.tensor(0.0, device=device)
        return embed_loss
    
    def to(self, device):
        """Move to device."""
        return self


class v8JDELoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8JDELoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # JDE() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4 + m.embed_dim
        self.embed_dim = m.embed_dim    # embedding dimension
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0, use_tags=True)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.embed_loss = MetricLearningLoss().to(device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, embed
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores, pred_embeds = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc, self.embed_dim), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_embeds = pred_embeds.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"], batch["tags"].view(-1, 1)), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes, gt_tags = targets.split((1, 4, 1), 2)  # cls, xyxy, tag
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _, target_tags = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            gt_tags
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

            # Embedding loss
            # Select the predicted embeddings for the foreground masks and the corresponding target tags
            pred_embeds = pred_embeds[fg_mask]  # (batch_fg_objects, embed_dim)
            target_tags = target_tags[fg_mask]  # (batch_fg_objects, 1)
            confidences = pred_scores[fg_mask].sigmoid().view(-1)   # (batch_fg_objects,)
            loss[3] = self.embed_loss(pred_embeds, target_tags, confidences)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.clr  # contrastive embedding gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, embed)
