import torch

from util.open_clip_util import TE_IE_Encoder

from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .matcher import build_matcher
from .model import CLSOVDETR
from .post_process import PostProcessSegm, PostProcessSIM
from .segmentation import DETRsegm
from .set_criterion import SetCriterion

def build_model(args):
    if args.dataset_file == "coco":
        num_classes = 91
    elif args.dataset_file == "lvis":
        num_classes = 1204
    else:
        raise NotImplementedError

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    TE_Encoder = TE_IE_Encoder(name=args.clip_name, pretrain=args.clip_pretrain, 
                               prompt_ensemble_type=args.prompt_ensemble_type,lvis=args.lvis)

    text_features = TE_Encoder.txt_features
    text_dim = TE_Encoder.txt_dim
    
    model = CLSOVDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        text_features=text_features,
        txt_dim=text_dim,
        freeze_feature_extractor=args.freeze_feature_extractor,
    )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(text_features, args)
    weight_dict = {"loss_feat_align": args.feature_loss_coef, "loss_bbox": args.bbox_loss_coef, "loss_ce": args.cls_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["boxes", "alignment", "labels"]
    if args.masks:
        losses = ["labels", "boxes", "masks"] 

    regional_feats = torch.load(args.guidance_feats_path)

    criterion =SetCriterion(
        matcher,
        weight_dict,
        losses,
        regional_feats,
        args.temperature,
        args.focal_alpha,
    )
    postprocessors = {"bbox": PostProcessSIM(text_features, args.temperature, args.lvis)} # PostProcess(text_features, args.temperature)
    criterion.to(device)

    if args.masks:
        postprocessors["segm"] = PostProcessSegm()

    return model, criterion, postprocessors
