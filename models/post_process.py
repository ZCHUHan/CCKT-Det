import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops

class PostProcessSIM(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""
    def __init__(self,
                 txt_emb,
                 temperature = 0.07,
                 lvis = False):
        super().__init__()

        self.temperature = temperature
        self.num_select = 300 if lvis else 100
        idx2label = {k: label for k, label in enumerate(txt_emb.keys())}
        self.idx2label_tensor = torch.tensor([idx2label[i] for i in range(len(idx2label))])
        self.txt_emb = torch.cat(list(txt_emb.values()), dim=0) # convert dict into [num_class, h_dim]

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        bs, query_num, _ = outputs["pred_boxes"].size()
        out_feats, out_bbox = outputs["pred_embed"], outputs["pred_boxes"]

        assert len(out_feats) == len(target_sizes) # bs
        assert target_sizes.shape[1] == 2

        out_emb = nn.functional.normalize(out_feats.flatten(0, 1), dim=-1) # [batch_size * num_queries, h_dim]
        similarity_score = torch.matmul(out_emb, self.txt_emb.T)
        sim_prob = torch.sigmoid(similarity_score / self.temperature).view(bs, query_num, -1) # select before recover the original shape 

        
        topk_values, topk_indexes = torch.topk(sim_prob.view(bs, -1), self.num_select, dim=1) 
        scores = topk_values
        topk_boxes = topk_indexes // self.txt_emb.shape[0]
        labels = topk_indexes % self.txt_emb.shape[0]
        labels = self.idx2label_tensor.to(out_feats.device)[labels]


        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"]
        outputs_masks = F.interpolate(
            outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
        )
        outputs_masks = outputs_masks.sigmoid() > self.threshold

        for i, (cur_mask, t, tt) in enumerate(
            zip(outputs_masks, max_target_sizes, orig_target_sizes)
        ):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results

