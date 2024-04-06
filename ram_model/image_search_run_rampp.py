import torch, pickle, argparse, re
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram_openset as inference

from ram.utils import build_openset_llm_label_embedding
from torch import nn
import json
import numpy as np
from tqdm import tqdm
from tardataset import TarDataset
import pickle
import os
import pandas as pd

def custom_collate_fn(batch):
    images = []
    jsons = []
    filepaths = []
    for i in batch:
        images.append(i[0])
        jsons.append(i[1])
        filepaths.append(i[2])
    return images, jsons, filepaths

def convert_to_rgb(image):
    return image.convert("RGB")

def run_rampp(args):
    # Make dataloader
    transforms = Compose([convert_to_rgb, Resize((args.image_size, args.image_size)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # special case for yfcc15m
    if args.pt_dataset == 'yfcc15m':
        # there are 15 folders with 99 tars each
        csv = pd.read_csv(os.path.join(args.load_path, 'yfcc15m_final_split_pageandimageurls.csv'))
        url_to_index_dict = csv.reset_index().set_index('url')['index'].to_dict()
        store_dir = args.chunk_idx // 99
        chunk_idx = args.chunk_idx % 99
        chunk = str(chunk_idx).zfill(5)
        save_chunk = str(args.chunk_idx).zfill(5)
        dataset = TarDataset(args.load_path+'/images_{}'.format(store_dir+1)+'/'+str(chunk)+'.tar', transform=transforms)
    elif args.pt_dataset == 'synthci30m':
        chunk = str(args.chunk_idx)
        save_chunk = chunk
        dataset = TarDataset(args.load_path+'/'+chunk+'.tar', transform=transforms, assume_jsons=False)
    elif args.pt_dataset == 'laion_aesthetics':
        chunk = str(args.chunk_idx).zfill(5)
        save_chunk = chunk
        dataset = TarDataset(args.load_path+'/'+chunk+'/00000.tar', transform=transforms, assume_jsons=False)
    else:
        chunk = str(args.chunk_idx).zfill(5)
        save_chunk = chunk
        dataset = TarDataset(args.load_path+'/'+chunk+'.tar', transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=custom_collate_fn, batch_size=args.batch_size_rampp, num_workers=4, shuffle=False, pin_memory=True)

    # Load model
    tagset = {}
    outs = []
    model = ram_plus(pretrained=args.pretrained, image_size=args.image_size, vit='swin_l')
    print('Building tag embedding:')
    with open(args.class_jsons, 'rb') as fo:
        descriptions = json.load(fo)

    if os.path.exists(os.path.join(args.features_dir, 'rampp_categories.pkl')):
        o_ = pickle.load(open(os.path.join(args.features_dir, 'rampp_categories.pkl'), 'rb'))
        openset_label_embedding, openset_categories = o_[0].cuda(), o_[1]
    else:
        openset_label_embedding, openset_categories = build_openset_llm_label_embedding(descriptions, args.cache_dir)
        pickle.dump([openset_label_embedding.cpu(), openset_categories], open(os.path.join(args.features_dir, 'rampp_categories.pkl'), 'wb'))

    model.tag_list = np.array(openset_categories)
    model.label_embed = nn.Parameter(openset_label_embedding.float())
    model.num_class = len(openset_categories)
    model.class_threshold = torch.ones(model.num_class) * args.confidence_threshold
    model = model.eval().cuda()

    # filenames is the name of the image
    # tags is the outputted tags at current confidence threshold
    # probs is the output probs per category (i.e. an np vector of 4029 values)
    outs = {'confidence_threshold': args.confidence_threshold, 'filenames': [], 'tags': [], 'probs': []}

    with torch.inference_mode(), torch.cuda.amp.autocast():
        for (image, json_file, filename) in tqdm(dataloader, ascii=True, total=len(dataloader)):
            if isinstance(image, list):
                image = torch.stack(image, dim=0).cuda(non_blocking=True)
            else:
                image = image.cuda(non_blocking=True)
            res, logits = inference(image, model, return_logits=True)
            probs = logits.half().cpu().numpy()
            assert(len(res) == len(filename))
            assert(len(probs) == len(filename))
            if args.pt_dataset == 'yfcc15m':
                for f, j, tag, prob in zip(filename, json_file, res, probs):
                    u = j['url']
                    row_num = url_to_index_dict[u]
                    outs['filenames'].append(row_num)
                    outs['tags'].append(tag)
                    outs['probs'].append(prob)
            else:
                for f, tag, prob in zip(filename, res, probs):
                    outs['filenames'].append(f)
                    outs['tags'].append(tag)
                    outs['probs'].append(prob)

    ### uncomment for qualitative visualisation ###
    # for ind_, (f, t) in enumerate(zip(outs['filenames'], outs['tags'])):
    #     print(ind_)
    #     print(f)
    #     print(t)
    #     print('----------------')
    #     if ind_==10:
    #         break

    # Save Outputs
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'rampp_outputs'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'rampp_outputs', '{}'.format(args.pt_dataset)), exist_ok=True)
    save_file(content=outs, filename=args.results_dir+'/rampp_outputs/{}/rampp_output_{}_{}'.format(args.pt_dataset, args.pt_dataset, save_chunk))
    return tagset

# Utils
def save_file(content, filename):
    with open(f'{filename}.pkl', "wb") as output_file:
        pickle.dump(content, output_file)

def load_file(filename):
    with open(f'{filename}.pkl', "rb") as output_file:
        content = pickle.load(output_file)
    return content

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dataset", type=str, default='cc3m')
    parser.add_argument("--load_path", type=str, default='../data/laion400m')
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--class_jsons", type=str, default='../gpt_descriptions/rampp_overall.json')
    parser.add_argument("--batch_size_rampp", type=int, default=64)
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="Confidence threshold for the RAM++ model")
    parser.add_argument('--pretrained', default='~/.cache/ram_plus_swin_large_14m.pth', help='Path to pretrained RAM++ model')
    parser.add_argument('--image_size', default=384, type=int, help='input image size (default: 448)')
    parser.add_argument('--cache_dir', type=str, help='cache_dir to save/load CLIP model weights')
    parser.add_argument('--features_dir', type=str, help='dir to save/load CLIP encoded test set features')
    parser.add_argument('--results_dir', type=str, help='dir to save/load results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(f'==> Args for this experiment are: {args}')

    tagset = run_rampp(args=args)
