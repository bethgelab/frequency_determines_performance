import os
import numpy as np
import torch
import open_clip
from tqdm import tqdm
import argparse
from downstream_dataloader import DataLoader
from zero_shot_templates import ensemble_templates
import json
import torchvision.transforms as transforms

from torchvision.utils import save_image

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
    ('RN50', 'openai'): ('RN50', 'openai'),

    # RN101
    ('RN101', 'yfcc15m'): ('RN101', 'yfcc15m'),
    ('RN101', 'openai'): ('RN101', 'openai'),

    # ViT-B-32
    ('ViT-B-32', 'openai'): ('ViT-B-32', 'openai'),
    ('ViT-B-32', 'laion400m'): ('ViT-B-32', 'laion400m_e32'),
    ('ViT-B-32', 'laion2b'): ('ViT-B-32', 'laion2b_s34b_b79k'),
    ('ViT-B-32', 'metaclip400m'): ('ViT-B-32-quickgelu', 'metaclip_400m'),
    ('ViT-B-32', 'metaclip2.5b'): ('ViT-B-32-quickgelu', 'metaclip_fullcc'),
    ('ViT-B-32', 'datacomp'): ('ViT-B-32', 'datacomp_xl_s13b_b90k'),
    ('ViT-B-32', 'laion200m_train_test_sim_normalized'): ('ViT-B-32', 'clip_ViT-B-32_train-test-sim-normalized_from_prasanna.pt'), # pretraining here just refers to the name of the checkpoint
    ('ViT-B-32', 'commonpool'): ('ViT-B-32', 'commonpool_m_laion_s128m_b4k'),

    # ViT-B-16
    ('ViT-B-16', 'openai'): ('ViT-B-16', 'openai'),
    ('ViT-B-16', 'laion400m'): ('ViT-B-16', 'laion400m_e32'),
    ('ViT-B-16', 'laion2b'): ('ViT-B-16', 'laion2b_s34b_b88k'),
    ('ViT-B-16', 'metaclip400m'): ('ViT-B-16-quickgelu', 'metaclip_400m'),
    ('ViT-B-16', 'metaclip2.5b'): ('ViT-B-16-quickgelu', 'metaclip_fullcc'),
    ('ViT-B-16', 'datacomp'): ('ViT-B-16', 'datacomp_xl_s13b_b90k'),
    ('ViT-B-16', 'cc12m'): ('ViT-B-16', 'clip_ViT-B-16_CC12M_from_slip.pt'), # pretraining here just refers to the name of the checkpoint
    ('ViT-B-16', 'cc3m'): ('ViT-B-16', 'clip_ViT-B-16_CC3M_from_slip.pt'), # pretraining here just refers to the name of the checkpoint
    ('ViT-B-16', 'yfcc15m'): ('ViT-B-16', 'clip_ViT-B-16_YFCC15M_from_slip.pt'), # pretraining here just refers to the name of the checkpoint
    ('ViT-B-16', 'synthci30m'): ('ViT-B-16', 'clip_ViT-B-16_SynthCI30M.pt'), # pretraining here just refers to the name of the checkpoint
    ('ViT-B-16', 'siglip'): ('hf-hub:timm/ViT-B-16-SigLIP', None),
    ('ViT-B-16', 'dfn'): ('ViT-B-16', 'dfn2b'),
    ('ViT-B-16', 'commonpool'): ('ViT-B-16', 'commonpool_l_laion_s1b_b8k'),

    # ViT-L-14
    ('ViT-L-14', 'openai'): ('ViT-L-14', 'openai'),
    ('ViT-L-14', 'laion400m'): ('ViT-L-14', 'laion400m_e32'),
    ('ViT-L-14', 'laion2b'): ('ViT-L-14', 'laion2b_s32b_b82k'),
    ('ViT-L-14', 'metaclip400m'): ('ViT-L-14-quickgelu', 'metaclip_400m'),
    ('ViT-L-14', 'metaclip2.5b'): ('ViT-L-14-quickgelu', 'metaclip_fullcc'),
    ('ViT-L-14', 'datacomp'): ('ViT-L-14', 'datacomp_xl_s13b_b90k'),
    ('ViT-L-14', 'commonpool'): ('ViT-L-14', 'commonpool_xl_laion_s13b_b90k'),

    # ViT-H-14
    ('ViT-H-14', 'laion2b'): ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-H-14', 'metaclip2.5b'): ('ViT-H-14-quickgelu', 'metaclip_fullcc'),
    ('ViT-H-14', 'dfn'): ('ViT-H-14-quickgelu', 'dfn5b'),

    # ViT-g-14
    ('ViT-g-14', 'laion2b'): ('ViT-g-14', 'laion2b_s34b_b88k'),

    # ViT-bigG-14
    ('ViT-bigG-14', 'laion2b'): ('ViT-bigG-14', 'laion2b_s39b_b160k'),

    # Siglip-specific archs
    ('SO400M', 'siglip'): ('hf-hub:timm/ViT-SO400M-14-SigLIP', None),
    ('ViT-L-16', 'siglip'): ('hf-hub:timm/ViT-L-16-SigLIP-256', None),

}

def get_model_metadata(args):
    if (args.backbone, args.pretraining) not in BACKBONES_AND_PRETRAINED:
        raise ValueError('({}, {}) is not a valid combination of backbone and pretraining dataset'.format(args.backbone, args.pretraining))
    else:
        return BACKBONES_AND_PRETRAINED[(args.backbone, args.pretraining)]

def construct_text_classifier(args, classnames, model, tokenizer):
    dataset = args.dataset
    if args.text_prompts == 'ensemble':
        prompt_templates = ensemble_templates[dataset]
    elif args.text_prompts == 'class':
        prompt_templates = ["{c}"]
    elif args.text_prompts == 'simple':
        prompt_templates = ["A photo of a {c}."]

    with torch.inference_mode(), torch.cuda.amp.autocast():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.replace("{c}", classname) for template in prompt_templates]  # format with class
            texts = tokenizer(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # L2 normalise text embedding
            class_embedding = class_embeddings.mean(dim=0) # take mean over all text embeddings for all prompts
            class_embedding /= class_embedding.norm() # L2 normalise mean embedding
            zeroshot_weights.append(class_embedding)
        # create shape CxN
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def construct_test_features(loader, model):
    test_features = []
    test_labels = []

    with torch.inference_mode(), torch.cuda.amp.autocast():
        for i, (images, target) in enumerate(tqdm(loader, ascii=True, total=len(loader))):
            images = images.cuda()
            target = target.cuda()
            # encode image
            image_features = model.encode_image(images)
            # L2 norm image embedding
            image_features /= image_features.norm(dim=-1, keepdim=True)
            test_features.append(image_features)
            test_labels.append(target)
    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)

    return test_features, test_labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to test on')
    parser.add_argument('--backbone', type=str, default='ViT-B-32', choices=['RN50', 'RN101', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14', 'ViT-bigG-14', 'ViT-L-16', 'SO400M'], help='CLIP backbone to use')
    parser.add_argument('--pretraining', type=str, default='laion400m', choices=['laion400m', 'laion2b', 'yfcc15m', 'metaclip400m', 'metaclip2.5b', 'cc3m', 'cc12m', 'datacomp', 'synthci30m', 'laion200m_train_test_sim_normalized', 'siglip', 'openai', 'commonpool', 'dfn'], help='CLIP pretraining dataset to use')
    parser.add_argument('--text_prompts', type=str, default='ensemble', choices=['ensemble', 'class', 'simple'])
    parser.add_argument('--cache_dir', type=str, help='cache_dir to save/load CLIP model weights')
    parser.add_argument('--features_dir', type=str, help='dir to save/load CLIP encoded test set features')
    parser.add_argument('--results_dir', type=str, help='dir to save/load results')
    args = parser.parse_args()

    assert args.features_dir is not None, 'features_dir cannot be None'
    assert args.results_dir is not None, 'results_dir cannot be None'

    # dummy parameters for dataloader
    args.val_batch_size = 256
    args.train_batch_size = 256
    args.k_shot = 1

    # load CLIP model
    backbone, pretrained = get_model_metadata(args)
    tokenizer = None
    if args.pretraining not in ['cc3m', 'cc12m', 'yfcc15m', 'synthci30m', 'laion200m_train_test_sim_normalized']:
        # load from open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, cache_dir=args.cache_dir)
    elif args.pretraining == 'siglip':
        # load from open_clip
        model, preprocess = open_clip.create_model_from_pretrained(pretrained, cache_dir=args.cache_dir)
        tokenizer = open_clip.get_tokenizer(pretrained)
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

    # load dataset
    data_loader = DataLoader(args, preprocess)
    _, _, test_images, test_loader, num_classes, string_classnames = data_loader.load_dataset()

    # load/save text classifier features
    text_classifier_path = os.path.join(args.features_dir, '{}_{}_{}_{}_text_classifier_features.pt'.format(args.dataset, args.backbone, args.pretraining, args.text_prompts))

    if os.path.exists(text_classifier_path):
        print('Loading text classifier weights...')
        text_classifier = torch.load(text_classifier_path)
    else:
        print('Creating and saving text classifier weights...')
        text_classifier = construct_text_classifier(args, string_classnames, model, tokenizer)
        torch.save(text_classifier, text_classifier_path)

    # load/save test set features
    image_feats_path = os.path.join(args.features_dir, '{}_{}_{}_test_features.pt'.format(args.dataset, args.backbone, args.pretraining))
    image_labels_path = os.path.join(args.features_dir, '{}_{}_{}_test_labels.pt'.format(args.dataset, args.backbone, args.pretraining))

    if os.path.exists(image_feats_path) and os.path.exists(text_classifier_path):
        print('Loading test features and labels...')
        test_features = torch.load(image_feats_path)
        test_labels = torch.load(image_labels_path)
    else:         
        print('Creating and saving test features and labels...')
        test_features, test_labels = construct_test_features(test_loader, model)
        torch.save(test_features, image_feats_path)
        torch.save(test_labels, image_labels_path)

    # accuracy computation
    logits = 100. * test_features @ text_classifier
    labels = test_labels
    np_preds = torch.argmax(logits, dim=1).cpu().numpy()
    np_labels = labels.cpu().numpy()   

    # Overall zero-shot accuracy
    zs_acc = 100 * (np_preds == np_labels).sum() / np_labels.shape[0]
    print('---------------------------------')
    print('ZS Acc for Dataset: {}, Backbone: {}, Pretrained: {}, Text-prompts: {} == '.format(args.dataset, args.backbone, args.pretraining, args.text_prompts), zs_acc)

    # Per-class accuracies
    per_class_accuracies = {}
    for idx, classname in enumerate(string_classnames):
        class_indices = np.where(np_labels == idx)
        class_preds = np_preds[class_indices]
        class_true = np_labels[class_indices]
        class_acc = 100 * (class_preds == class_true).sum() / class_true.shape[0]
        per_class_accuracies[classname] = class_acc

    print('Per-class accuracies: {}'.format(per_class_accuracies))
    print('---------------------------------')

    # dump results
    all_results = {'full': zs_acc, 'classwise': per_class_accuracies}
    results_path = os.path.join(args.results_dir, '{}_{}_results.json'.format(args.dataset, args.text_prompts))
    os.makedirs(args.results_dir, exist_ok=True)

    if os.path.exists(results_path):
    	with open(results_path, 'r') as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
    output_dict['backbone{}_pretrained{}'.format(args.backbone, args.pretraining)] = all_results

    with open(results_path, 'w') as f:
        json.dump(output_dict, f, indent=4)
