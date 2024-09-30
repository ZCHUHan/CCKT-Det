import torch
import copy

from PIL import Image, ImageOps
from tqdm import tqdm
import json
from collections import defaultdict

from util.open_clip_util import TE_IE_Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
encoder = TE_IE_Encoder(lvis=True)
encoder.to(device)

# Json and COCO dataset dir path
anno_path = 'data/lvis/lvis_v1_train_norare.json' #'data/coco/annotations/instances_train2017_seen_2_proposal.json'
raw_img_file_dir = "data/coco/train2017/"
save_path = "data/lvis/train2017_regional_feats.pkl"

with open(anno_path, "r") as f:
    data = json.load(f)

img2ann_gt = defaultdict(list)
for temp in data['annotations']:
    img2ann_gt[temp['image_id']].append(temp)

dic = {}
for image_id in tqdm(img2ann_gt.keys()):
    file_name = raw_img_file_dir + f"{image_id}".zfill(12) + ".jpg"
    image = Image.open(file_name).convert("RGB")
    
    for value in img2ann_gt[image_id]:
        ind = value['id']
        bbox = copy.deepcopy(value['bbox'])
        if (bbox[1] < 16) or (bbox[2] < 16):
            continue
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        roi = encoder.clip_preprocess(image.crop(bbox)).to(device).unsqueeze(0)
        with torch.no_grad():
            roi_features = encoder.clip_model.encode_image(roi)
        
        category_id = value['category_id']

        if category_id in dic.keys():
            dic[category_id].append(roi_features)
        else:
            dic[category_id] = [roi_features]


for key in dic.keys():
    dic[key] = torch.cat(dic[key], 0)

print("dic", dic.keys())
sorted_dict = dict(sorted(dic.items()))

# sort is to ensure each idx corresponds to label correctly
print("dic", sorted_dict.keys())
torch.save(sorted_dict, save_path)
