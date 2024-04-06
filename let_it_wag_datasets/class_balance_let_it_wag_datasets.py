import os
from tqdm import tqdm
import json
import shutil
import random

let_it_wag_original_data_path = '/path/to/dataset/'

class_folders = [os.path.join(let_it_wag_original_data_path, x) for x in os.listdir(let_it_wag_original_data_path) if os.path.isdir(os.path.join(let_it_wag_original_data_path, x))]

data_stats = {}

for class_dir in tqdm(class_folders, total=len(class_folders), ascii=True):
    img_files = [x for x in os.listdir(class_dir) if '.jpg' in x]
    if len(img_files) > 450:
        data_stats[class_dir.split('/')[-1]] = len(img_files)

sorted_dict = {k: v for k, v in sorted(data_stats.items(), key=lambda item: -1*item[1])}

# do final transfer
final_path = '/path/to/final/dataset'
os.makedirs(final_path, exist_ok=True)

for k in tqdm(sorted_dict, ascii=True, total=len(sorted_dict)):
    orig_dir = os.path.join(let_it_wag_original_data_path, k)
    
    # get all images
    img_list = [os.path.join(orig_dir, x) for x in os.listdir(orig_dir)]

    # randomly shuffle the images
    sampled_images = random.sample(img_list, 450)
    
    # for each image, copy it to the final path
    dest_path = os.path.join(final_path, k)
    os.makedirs(dest_path, exist_ok=True)
    for img in tqdm(sampled_images, ascii=True, total=len(sampled_images)):
        dest = os.path.join(dest_path, img.split('/')[-1])
        shutil.copy(img, dest)