import os
import regex as re
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from scipy.linalg import sqrtm
import pickle

# Load DINOV2 model
print("Loading DINOV2 model")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14.eval()

# Define a function to extract DINOV2 features from images
def extract_dinov2_features(image_paths, image_transform):

    features = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = image_transform(img).unsqueeze(0)
        with torch.no_grad():
            feature = dinov2_vits14(img)
        features.append(feature.squeeze().cpu().numpy())
    return np.array(features)

def calculate_fid_score(real_features, fake_features):
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    cov_sqrt = sqrtm(sigma_real.dot(sigma_fake))

    # Handling numerical instability
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # Compute FID score
    fid_score = np.dot(diff, diff)+ np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    # fid_score = ssdiff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return fid_score



feat_real_root = '/Users/heikekoenig/irp/blind_name_only_transfer/let_it_wag_concepts/features/common/'
feat_gen_root = '/Users/heikekoenig/irp/blind_name_only_transfer/let_it_wag_concepts/t2i_features/dreamlike/common/'
fid_save = f'/Users/heikekoenig/irp/blind_name_only_transfer/let_it_wag_concepts/results/fid/{feat_gen_root.split("/")[-3]}_{feat_gen_root.split("/")[-2]}.pkl'

fid = {}
for file in sorted(os.listdir(feat_gen_root)):
    #load generated features
    print("loading features for ", file.split('.')[0])

    #remove ignored concepts
    if not os.path.isfile(os.path.join(feat_real_root, file)):
        print("Ignoring ", file)
        continue

    generated_features = np.load(os.path.join(feat_gen_root, file))

    #load real features
    real_features = np.load(os.path.join(feat_real_root, file))

    # Calculate FID score
    # score = []
    # for i in range(len(generated_features)):
    #     fid_score = 0.0
    #     for j in range(len(real_features)):
    #         fid_score = fid_score + calculate_fid_score2(real_features[j], generated_features[i])
    #     score.append(fid_score/len(real_features))
    # fid_score_avg = np.mean(score)

    fid_score_avg = calculate_fid_score(real_features, generated_features)
    fid[file.split('.')[0]] = fid_score_avg
    print("FID score for ", file.split('.')[0], " is ", fid_score_avg)

print("Saving FID scores")
with open(fid_save, 'wb') as f:
    pickle.dump(fid, f)