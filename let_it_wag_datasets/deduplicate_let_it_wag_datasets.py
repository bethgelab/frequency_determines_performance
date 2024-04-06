import os
from imagededup.methods import PHash, DHash, AHash, WHash
from tqdm import tqdm
import json

let_it_wag_data_path = '/path/to/dataset/'

class_folders = [os.path.join(let_it_wag_data_path, x) for x in os.listdir(let_it_wag_data_path) if os.path.isdir(os.path.join(let_it_wag_data_path, x))]

phasher = PHash(verbose=False)
threshold = 10

remove_images_dict = {}

for class_dir in tqdm(class_folders, total=len(class_folders), ascii=True):

	duplicates = phasher.find_duplicates(image_dir=class_dir, scores=False, max_distance_threshold=threshold)

	remove_images = set()
	for img in duplicates:
		if img in remove_images:
			continue
		if len(duplicates[img]) > 0:
			remove_images.update(duplicates[img])

	print('Number of duplicates: {}'.format(len(list(remove_images))))
	print('Total num images: {}'.format(len(os.listdir(class_dir))))

	remove_images_dict[class_dir.split('/')[-1]] = list(remove_images)

	with open('let_it_wag_fg_duplicate_indices_phash.json', 'w') as f:
		json.dump(remove_images_dict, f, indent=4)