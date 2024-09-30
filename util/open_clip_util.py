import torch
from torch import nn
from torch.nn import functional as F


import logging


logging.basicConfig(level=logging.DEBUG, filename='debug.log', filemode='w', format='%(message)s')
logger = logging.getLogger()


import open_clip
from util.misc import NestedTensor
from util.prompt_template import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES
from util.coco_class_name import COCO_CLASS_NAME, OVD_SETTING_ALL_IDS
from util.lvis_class_name import LVIS_CLASS_NAME
from typing import Dict, List


class TE_IE_Encoder(nn.Module):
    def __init__(
        self,
        name='ViT-L-14', # ('ViT-H-14', 'laion2b_s32b_b79k')
        pretrain='laion2b_s32b_b82k',
        prompt_ensemble_type="imagenet_select", # "single", "imagenet" "imagenet_select"
        lvis=False,
    ):
        """
        Args:
            
        """
        super().__init__()

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name=name, 
            pretrained=pretrain,
            )
        self.tokenizer = open_clip.get_tokenizer(name)


        # prepare txt encoder and txt guidance
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            self.prompt_templates = IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            self.prompt_templates = IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            self.prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.lvis = lvis
        self.txt_features = self.build_txt_embeddings(categories=LVIS_CLASS_NAME) if lvis else self.build_txt_embeddings() 
        self.idx2label = {k: label for k, label in enumerate(self.txt_features.keys())}

        # use clip config
        self.txt_dim = open_clip.get_model_config(model_name=name).get('embed_dim')
        self.temperature = 0.07

        # pseudo label hyperparam
        self.pseudo_label_threshold = 0.75 if lvis else 0.80 # 0.8 for coco
        self.ins_fix_num_per_image = 12 if lvis else 6 # 6 for coco

    @torch.no_grad()
    def build_txt_embeddings(self, categories=COCO_CLASS_NAME): 
        run_on_gpu = torch.cuda.is_available()
        clip_model = self.clip_model.to('cuda' if run_on_gpu else 'cpu')
        
        zeroshot_weights = []
        for _, category in categories.items():
            texts = [
                template.format(category)
                for template in self.prompt_templates
            ]
            # tokenize
            texts = self.tokenizer(texts)

            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = clip_model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            zeroshot_weights.append(text_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        if run_on_gpu:
            zeroshot_weights = zeroshot_weights.cuda()
        
        zeroshot_weights = zeroshot_weights.t().float() # (class_num, 768 txt_dim)
        
        # cache as dict form: {coco_label_idx: txt_feat_weights}
        feat_dict: Dict[int, torch.Tensor] = {}
        if self.lvis:
            feat_dict = {i+1: weight.unsqueeze(0) for i, weight in enumerate(zeroshot_weights)}
        else:
            feat_dict = {i+1: weight.unsqueeze(0) for i, weight in enumerate(zeroshot_weights) if i+1 in OVD_SETTING_ALL_IDS}
        return feat_dict
    
    @torch.no_grad()
    def forward(self, image): 
        cls_weights = torch.cat(list(self.txt_features.values()), dim=0)

        image_features = self.clip_model.encode_image(image) # the first element is the pooled feature, bs, HxW, dim
        image_features /= image_features.norm(dim=-1, keepdim=True)

        #print("image_features", image_features.size(), "cls_weights", cls_weights.size())
        similarity_score = (image_features @ cls_weights.T) # multi-label cls, using sigmoid instead of softmax?
        probabilities = torch.sigmoid(similarity_score / self.temperature) # img_num, cls_num

        high_confidence_scores = probabilities[(probabilities > self.pseudo_label_threshold).nonzero(as_tuple=True)]   
        high_confidence_indices = (probabilities > self.pseudo_label_threshold).nonzero(as_tuple=True)[1]

        sorted_scores, sorted_indices = torch.sort(high_confidence_scores, descending=True)
        sorted_labels = high_confidence_indices[sorted_indices]

        pseudo_labels = [self.idx2label[idx] for idx in sorted_labels.tolist()]
        
        # ensure there always some pseudo labels TODO make a fixed length
        #print("sorted pseudo labels index", pseudo_labels)
        # ensure not empty
        if len(pseudo_labels)==0:
            pseudo_labels = [self.idx2label[torch.randint(low=0, high=len(self.idx2label), size=(1,)).item()]]
        if self.ins_fix_num_per_image>0:
            pseudo_labels = pseudo_labels[:self.ins_fix_num_per_image]
        
        return pseudo_labels
    