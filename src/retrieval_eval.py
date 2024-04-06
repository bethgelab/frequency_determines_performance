from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import open_clip
import os
from PIL import Image
from torchvision import transforms
import argparse
import pickle
from tqdm import tqdm
import pandas as pd
import json

MODEL_AND_BACKBONE = {
    ('clip', 'ViT-B-32'): ('ViT-B-32', 'openai'),
    ('clip', 'ViT-bigG-14'): ('ViT-bigG-14', 'laion2b_s39b_b160k'),
}

# Notes: 
# CyCLIP (https://github.com/goel-shashank/CyCLIP/tree/main) has CLIP-RN50 models trained on CC3M
# Quality-Not_Quantity (https://github.com/mlfoundations/clip_quality_not_quantity) has CLIP-RN50 models (released here: https://huggingface.co/thaottn/) on CC12M/YFCC15M
# OpenCLIP has CLIP-RN50 models on YFCC15M 
# SLIP (https://github.com/facebookresearch/SLIP) has CLIP-ViT-B/L versions on YFCC15M and CLIP-ViT-B versions on CC3M/CC12M
# [not used, because they use a different yfcc15m split] DeCLIP (https://github.com/Sense-GVT/DeCLIP) has CLIP-ViT-B/32 and CLIP-RN50 models trained on YFCC15M 
# SynthCLIP (https://github.com/hammoudhasan/SynthCLIP) has CLIP-ViT-B-16 trained on SynthCI-30M (for the metaclip style analysis)
# Train-test similarity normalised from P. Mayilvahanan et al, "https://arxiv.org/abs/2310.09562"
BACKBONES_AND_PRETRAINED = {
    # RN50
    ('RN50', 'cc3m'): ('RN50', 'clip_RN50_CC3M_from_cyclip.pt'), # pretraining here just refers to the name of the checkpoint
    ('RN50', 'cc12m'): ('RN50', 'hf-hub:thaottn/OpenCLIP-resnet50-CC12M'), # pretraining here refers to the HF-hub name of the checkpoint
    ('RN50', 'yfcc15m'): ('RN50', 'hf-hub:thaottn/OpenCLIP-resnet50-YFCC15M'), # pretrining here refers to the HF-hub name of the checkpoint

    # RN101
    ('RN101', 'yfcc15m'): ('RN101', 'yfcc15m'),

    # ViT-B-32
    ('ViT-B-32', 'laion400m'): ('ViT-B-32', 'laion400m_e32'),
    ('ViT-B-32', 'laion2b'): ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-B-32', 'metaclip400m'): ('ViT-B-32-quickgelu', 'metaclip_400m'),
    ('ViT-B-32', 'metaclip2.5b'): ('ViT-B-32', 'metaclip_fullcc'),
    ('ViT-B-32', 'datacomp'): ('ViT-B-32', 'datacomp_xl_s13b_b90k'),
    ('ViT-B-32', 'laion200m_train_test_sim_normalized'): ('ViT-B-32', 'clip_ViT-B-32_train-test-sim-normalized_from_prasanna.pt'), # pretraining here just refers to the name of the checkpoint

    # ViT-B-16
    ('ViT-B-16', 'laion400m'): ('ViT-B-16', 'laion400m_e32'),
    ('ViT-B-16', 'laion2b'): ('ViT-B-16', 'laion2b_s34b_b88k'),
    ('ViT-B-16', 'metaclip400m'): ('ViT-B-16', 'metaclip_400m'),
    ('ViT-B-16', 'metaclip2.5b'): ('ViT-B-16', 'metaclip_fullcc'),
    ('ViT-B-16', 'datacomp'): ('ViT-B-16', 'datacomp_xl_s13b_b90k'),
    ('ViT-B-16', 'cc12m'): ('ViT-B-16', 'clip_ViT-B-16_CC12M_from_slip.pt'), # pretraining here just refers to the name of the checkpoint
    ('ViT-B-16', 'cc3m'): ('ViT-B-16', 'clip_ViT-B-16_CC3M_from_slip.pt'), # pretraining here just refers to the name of the checkpoint
    ('ViT-B-16', 'yfcc15m'): ('ViT-B-16', 'clip_ViT-B-16_YFCC15M_from_slip.pt'), # pretraining here just refers to the name of the checkpoint
    ('ViT-B-16', 'synthci30m'): ('ViT-B-16', 'clip_ViT-B-16_SynthCI30M.pt'), # pretraining here just refers to the name of the checkpoint

    # ViT-L-14
    ('ViT-L-14', 'laion400m'): ('ViT-L-14', 'laion400m_e32'),
    ('ViT-L-14', 'laion2b'): ('ViT-L-14', 'laion2b_s32b_b82k'),
    ('ViT-L-14', 'metaclip400m'): ('ViT-L-14', 'metaclip_400m'),
    ('ViT-L-14', 'metaclip2.5b'): ('ViT-L-14', 'metaclip_fullcc'),
    ('ViT-L-14', 'datacomp'): ('ViT-L-14', 'datacomp_xl_s13b_b90k'),

    # ViT-H-14
    ('ViT-H-14', 'laion2b'): ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-H-14', 'metaclip2.5b'): ('ViT-H-14', 'metaclip_fullcc'),

    # ViT-g-14
    ('ViT-g-14', 'laion2b'): ('ViT-g-14', 'laion2b_s34b_b88k'),

    # ViT-bigG-14
    ('ViT-bigG-14', 'laion2b'): ('ViT-bigG-14', 'laion2b_s39b_b160k'),
}

def get_model_metadata(args):
    if (args.backbone, args.pretraining) not in BACKBONES_AND_PRETRAINED:
        raise ValueError('({}, {}) is not a valid combination of backbone and pretraining dataset'.format(args.backbone, args.pretraining))
    else:
        return BACKBONES_AND_PRETRAINED[(args.backbone, args.pretraining)]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='coco', choices=['coco', 'flickr'])
    parser.add_argument('--backbone', type=str, default='ViT-B-32', choices=['RN50', 'RN101', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14', 'ViT-bigG-14'], help='CLIP backbone to use')
    parser.add_argument('--pretraining', type=str, default='laion400m', choices=['laion400m', 'laion2b', 'yfcc15m', 'metaclip400m', 'metaclip2.5b', 'cc3m', 'cc12m', 'datacomp', 'synthci30m', 'laion200m_train_test_sim_normalized'], help='CLIP pretraining dataset to use')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--features_dir', type=str, help='dir to save/load CLIP encoded test set features')
    parser.add_argument('--cache_dir', type=str, help='cache_dir to save/load CLIP model weights')
    parser.add_argument('--results_dir', type=str, help='results_dir to save/load results')
    args = parser.parse_args()
    return args

def data_load(args, preprocessor, tokenizer):
    if args.dataset == 'coco':
        dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")
    else:
        dataset = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval")
    data_split = dataset['test']

    def collate_fn(batch):
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        batch['image'] = torch.stack([preprocessor(b) for b in batch['image']], dim=0)
        # take only 5 captions of coco per image: refer https://github.com/openai/CLIP/issues/115
        batch['text'] = torch.stack([tokenizer(b[:5]) for b in batch['caption']], dim=0)
        return batch

    data_loader = DataLoader(data_split, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    return data_split, data_loader

def get_embeddings(args, model, dataloader):
    # image_to_text_map[i] gives the corresponding text indices for the ith image
    #  (as there are multiple pieces of text for each image)
    image_to_text_map = []

    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []

    text_index = 0
    image_index = 0

    image_embeddings = []
    text_embeddings = []

    all_predictions = []
    labels = []
    preds = []

    # Now you can iterate over the DataLoader in your training loop
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, ascii=True, total=len(dataloader)):
            images = batch['image'].cuda()
            texts = batch['text'].cuda()

            # text has shape B x 5 x 77
            batch_size, captions_per_image, _ = texts.shape

            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1            

            # B x 5 x 77 -> (B*5) x 77
            texts = torch.flatten(texts, start_dim=0, end_dim=1)
            
            image_embeddings.append(model.encode_image(images))
            text_embeddings.append(model.encode_text(texts))

        image_embeddings = torch.cat(image_embeddings)
        text_embeddings = torch.cat(text_embeddings)
        text_to_image_map = torch.LongTensor(text_to_image_map).cuda()
        image_to_text_map = torch.LongTensor(image_to_text_map).cuda()

        # Normalise encodings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        return image_embeddings, text_embeddings, image_to_text_map, text_to_image_map

def evaluate(args, model, dataset, dataloader, concepts_to_text_samples_map, text_samples_to_concepts_map, concepts_to_image_samples_map, image_samples_to_concepts_map):
    image_embeddings, text_embeddings, image_to_text_map, text_to_image_map = get_embeddings(args, model, dataloader)

    num_text = text_embeddings.shape[0]
    num_im = image_embeddings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    # text-to-image recall
    print("Text-to-image recall...")

    dist_matrix = text_embeddings @ image_embeddings.T  # dist_matrix[i] gives logits for ith text

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    text_to_image_recall = []

    k_vals = [1, 5, 10]

    conceptwise_text_to_image_recall = {k: {} for k in k_vals}

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        assert correct.shape[0] == len(text_samples_to_concepts_map)

        for sample_index in range(correct.shape[0]):
            for concept in text_samples_to_concepts_map[sample_index]:
                if concept not in conceptwise_text_to_image_recall[k]:
                    conceptwise_text_to_image_recall[k][concept] = []
                conceptwise_text_to_image_recall[k][concept].append(int(correct[sample_index].item()))

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    save_text_to_image_metrics = {k:{} for k in k_vals}
    for k_ind, k in enumerate(k_vals):
        save_text_to_image_metrics[k] = {'full':text_to_image_recall[k_ind], 'conceptwise': {}}
        for concept in conceptwise_text_to_image_recall[k]:
            save_text_to_image_metrics[k]['conceptwise'][concept] = sum(conceptwise_text_to_image_recall[k][concept]) / len(conceptwise_text_to_image_recall[k][concept])

    # image-to-text recall
    print("Image-to-text recall...")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    image_to_text_recall = []

    conceptwise_image_to_text_recall = {k: {} for k in k_vals}

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        # For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        assert correct.shape[0] == len(image_samples_to_concepts_map)

        for sample_index in range(correct.shape[0]):
            for concept in image_samples_to_concepts_map[sample_index]:
                if concept not in conceptwise_image_to_text_recall[k]:
                    conceptwise_image_to_text_recall[k][concept] = []
                conceptwise_image_to_text_recall[k][concept].append(int(correct[sample_index].item()))

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)

    save_image_to_text_metrics = {k:{} for k in k_vals}
    for k_ind, k in enumerate(k_vals):
        save_image_to_text_metrics[k] = {'full':image_to_text_recall[k_ind], 'conceptwise': {}}
        for concept in conceptwise_image_to_text_recall[k]:
            save_image_to_text_metrics[k]['conceptwise'][concept] = sum(conceptwise_image_to_text_recall[k][concept]) / len(conceptwise_image_to_text_recall[k][concept])

    print("Done.")
    return text_to_image_recall, image_to_text_recall, save_image_to_text_metrics, save_text_to_image_metrics

if __name__ == '__main__':
    
    args = get_args()
    k_vals = [1, 5, 10]

    # Assert CUDA
    assert(torch.cuda.is_available())
    device = torch.device("cuda")

    # load CLIP model
    backbone, pretrained = get_model_metadata(args)
    tokenizer = None
    if args.pretraining not in ['cc3m', 'cc12m', 'yfcc15m', 'synthci30m', 'laion200m_train_test_sim_normalized']:
        # load from open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, cache_dir=args.cache_dir)
    elif backbone == 'RN101' and 'yfcc15m' in pretrained.lower():
        # load from open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, cache_dir=args.cache_dir)
    elif backbone=='RN50' and 'cc3m' in pretrained.lower():
        # load from cyclip
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained='openai', cache_dir=args.cache_dir)
        model_ckpt = {k.replace('module.', ''):v for k,v in dict(torch.load(os.path.join(args.cache_dir, pretrained), map_location='cuda')['state_dict']).items()}
        assert sorted(list(model.state_dict().keys())) == sorted(list(model_ckpt.keys()))
        model.load_state_dict(model_ckpt)
    elif backbone=='ViT-B-32' and 'train-test-sim-normalized' in pretrained.lower():
        # use open_clip load with modified checkpoint loading
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained='openai', cache_dir=args.cache_dir)
        model_ckpt = {k.replace('module.', ''):v for k,v in dict(torch.load(os.path.join(args.cache_dir, pretrained), map_location='cuda')['state_dict']).items()}
        assert sorted(list(model.state_dict().keys())) == sorted(list(model_ckpt.keys()))
        model.load_state_dict(model_ckpt)
    elif backbone == 'RN50' and 'yfcc15m' in pretrained.lower():
        # load from hf
        model, _, preprocess = open_clip.create_model_and_transforms(pretrained)
        tokenizer = open_clip.get_tokenizer(pretrained)
    elif backbone == 'RN50' and 'cc12m' in pretrained.lower():
        # load from hf
        model, _, preprocess = open_clip.create_model_and_transforms(pretrained)
        tokenizer = open_clip.get_tokenizer(pretrained)
    elif backbone=='ViT-B-16' and ('cc3m' in pretrained.lower() or 'cc12m' in pretrained.lower() or 'yfcc15m' in pretrained.lower() or 'synthci30m' in pretrained.lower()):
        # load from slip
        from slip import models as slip_models
        ckpt = torch.load(os.path.join(args.cache_dir, pretrained), map_location='cuda')
        model_args = ckpt['args']
        model_ckpt = {k.replace('module.', ''):v for k,v in dict(ckpt['state_dict']).items()}
        model = getattr(slip_models, model_args.model)(rand_embed=False,
            ssl_mlp_dim=model_args.ssl_mlp_dim, ssl_emb_dim=model_args.ssl_emb_dim)
        model.load_state_dict(model_ckpt, strict=True)
        preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: x.convert('RGB'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

    if tokenizer is None:
        tokenizer = open_clip.get_tokenizer(backbone)
    model.cuda()
    model.eval()

    preprocessor = preprocess
    print('Loaded {} model...'.format(args.backbone))

    # Get dataset
    dataset, dataloader = data_load(args, preprocessor, tokenizer)
    print('Loaded {} dataset...'.format(args.dataset))

    # Get sample-wise concepts
    with open(os.path.join(args.features_dir, '{}_concepts_to_text_samples_map.pkl'.format(args.dataset)), 'rb') as f:
        concepts_to_text_samples_map = pickle.load(f)
    with open(os.path.join(args.features_dir, '{}_text_samples_to_concepts_map.pkl'.format(args.dataset)), 'rb') as f:
        text_samples_to_concepts_map = pickle.load(f)
    with open(os.path.join(args.features_dir, '{}_concepts_to_image_samples_map.pkl'.format(args.dataset)), 'rb') as f:
        concepts_to_image_samples_map = pickle.load(f)
    with open(os.path.join(args.features_dir, '{}_image_samples_to_concepts_map.pkl'.format(args.dataset)), 'rb') as f:
        image_samples_to_concepts_map = pickle.load(f)

    # Evaluate model
    t2i, i2t, save_image_to_text_metrics, save_text_to_image_metrics = evaluate(args, model, dataset, dataloader, concepts_to_text_samples_map, text_samples_to_concepts_map, concepts_to_image_samples_map, image_samples_to_concepts_map)

    print("Text-to-image Recall@K")
    for k, x in zip(k_vals, t2i):
        print(f" R@{k}: {100*x:.2f}%")

    print("Image-to-text Recall@K")
    for k, x in zip(k_vals, i2t):
        print(f" R@{k}: {100*x:.2f}%")

    # dump results
    t2i_k1_results_path = os.path.join(args.results_dir, '{}_t2i_k=1_results.json'.format(args.dataset))
    t2i_k5_results_path = os.path.join(args.results_dir, '{}_t2i_k=5_results.json'.format(args.dataset))
    t2i_k10_results_path = os.path.join(args.results_dir, '{}_t2i_k=10_results.json'.format(args.dataset))

    i2t_k1_results_path = os.path.join(args.results_dir, '{}_i2t_k=1_results.json'.format(args.dataset))
    i2t_k5_results_path = os.path.join(args.results_dir, '{}_i2t_k=5_results.json'.format(args.dataset))
    i2t_k10_results_path = os.path.join(args.results_dir, '{}_i2t_k=10_results.json'.format(args.dataset))

    os.makedirs(args.results_dir, exist_ok=True)

    #### Save T2I results ####
    if os.path.exists(t2i_k1_results_path):
    	with open(t2i_k1_results_path, 'r') as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
    output_dict['backbone{}_pretrained{}'.format(args.backbone, args.pretraining)] = save_text_to_image_metrics[1]

    with open(t2i_k1_results_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

    if os.path.exists(t2i_k5_results_path):
    	with open(t2i_k5_results_path, 'r') as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
    output_dict['backbone{}_pretrained{}'.format(args.backbone, args.pretraining)] = save_text_to_image_metrics[5]

    with open(t2i_k5_results_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

    if os.path.exists(t2i_k10_results_path):
    	with open(t2i_k10_results_path, 'r') as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
    output_dict['backbone{}_pretrained{}'.format(args.backbone, args.pretraining)] = save_text_to_image_metrics[10]

    with open(t2i_k10_results_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

    #### Save I2T results ####
    if os.path.exists(i2t_k1_results_path):
    	with open(i2t_k1_results_path, 'r') as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
    output_dict['backbone{}_pretrained{}'.format(args.backbone, args.pretraining)] = save_image_to_text_metrics[1]

    with open(i2t_k1_results_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

    if os.path.exists(i2t_k5_results_path):
    	with open(i2t_k5_results_path, 'r') as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
    output_dict['backbone{}_pretrained{}'.format(args.backbone, args.pretraining)] = save_image_to_text_metrics[5]

    with open(i2t_k5_results_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

    if os.path.exists(i2t_k10_results_path):
    	with open(i2t_k10_results_path, 'r') as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
    output_dict['backbone{}_pretrained{}'.format(args.backbone, args.pretraining)] = save_image_to_text_metrics[10]

    with open(i2t_k10_results_path, 'w') as f:
        json.dump(output_dict, f, indent=4)