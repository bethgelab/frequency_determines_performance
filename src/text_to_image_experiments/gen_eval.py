import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import os
from PIL import Image
from scipy.linalg import sqrtm

# Load DINOV2 model
print("Loading DINOV2 model")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14.eval()

class WagDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, filename) for filename in sorted(os.listdir(root_dir)) if os.path.isfile(os.path.join(root_dir, filename))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

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

# Calculate FID score
def calculate_fid_score(real_features, fake_features):
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake

    # Calculate square root of covariance product
    cov_sqrt = sqrtm(sigma_real.dot(sigma_fake))

    # Handling numerical instability
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # Compute FID score
    fid_score = np.dot(diff, diff) + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return fid_score

# Example usage
wag_dir = '/Users/heikekoenig/irp/wag_fg/'
gen_dir = '/Users/heikekoenig/irp/blind_name_only_transfer/let_it_wag_concepts/images/sdxl/common/'
feat_save_root = '/Users/heikekoenig/irp/blind_name_only_transfer/let_it_wag_concepts/features/fg/'

if os.path.exists(os.path.join(gen_dir, '.DS_Store')):
    # Remove the file
    os.remove(os.path.join(gen_dir, '.DS_Store'))
generated_image_paths = [os.path.join(root, file) for root, dirs, files in sorted(os.walk(gen_dir)) for file in files]

print("Creating dataloader for real images")
image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Extract features from real images
print("Creating features for real images")

for concept in sorted(os.listdir(wag_dir)):
    print(concept)
    concept_name = concept.lower().replace(" ", "_")
    if os.path.isfile(os.path.join(feat_save_root, '{}.npy'.format(concept_name))):
        continue
    concept_dir = os.path.join(wag_dir, concept)

    custom_dataset = WagDataset(root_dir=concept_dir, transform=image_transform)

    # Define your custom dataloader
    dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

    features = []
    for batch in dataloader:
        with torch.no_grad():
            feature_batch = dinov2_vits14(batch)
        features.append(feature_batch.half().cpu())
    real_features = np.concatenate(features, axis =0)

    feat_save_path = os.path.join(feat_save_root,'{}.npy'.format(concept_name))
    np.save(feat_save_path, real_features)


print("Creating features for generated images")
fake_features = extract_dinov2_features(generated_image_paths, image_transform, 'generated')


# Calculate FID score
print("Calculating FID score")
fid_score = calculate_fid_score(real_features, fake_features)
print("FID Score:", fid_score)
