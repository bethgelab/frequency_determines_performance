import os
import json
import argparse
from tqdm import tqdm
import pickle
import numpy as np
from itertools import product

# Corrected function to generate all combinations of capitalized and non-capitalized versions
def generate_combinations(name):
    combinations_dict = {}
    words = name.split()
    # Generate all combinations of capitalized and non-capitalized versions for each word
    word_combinations = [[word.lower(), word.capitalize()] for word in words]
    all_combinations = list(product(*word_combinations))
    # Join each combination with underscores and add to the list for this name
    combinations = ['_'.join(combination) for combination in all_combinations]
    return combinations

def main(args):

    # dummy parameters for dataloader
    args.val_batch_size = 64
    args.train_batch_size = 256
    args.k_shot = 1
    args.dataset = args.downstream_dataset

    # get downstream classes
    if args.downstream_dataset not in ['coco', 'flickr', 't2idrawbench', 't2iparti', 't2icoco', 't2i']:
        res_folder = '../zero_shot_evaluations'
        string_classnames = list(json.load(open(os.path.join(res_folder, '{}_class_results.json'.format(args.downstream_dataset))))['backboneViT-B-16_pretrainedcc12m']['classwise'].keys())
        if args.downstream_dataset == 'cifar10':
            string_classnames[0] = 'airplane'
        downstream_classes = string_classnames
        if args.downstream_dataset == 'imagenet-r':
            downstream_classes = [d.replace('_', ' ') for d in downstream_classes]
    elif args.downstream_dataset in ['coco', 'flickr']:
        res_folder = '../retrieval_evaluations'
        string_classnames = list(json.load(open(os.path.join(res_folder, '{}_i2t_k=1_results.json'.format(args.downstream_dataset))))['backboneViT-B-16_pretrainedcc12m']['conceptwise'].keys())
        downstream_classes = string_classnames
        if args.downstream_dataset == 'coco':
            downstream_classes.remove('')
    elif args.downstream_dataset in ['t2icoco', 't2idrawbench', 't2iparti', 't2i']:
        res_folder = '../t2i_evaluations'
        string_classnames = list(json.load(open(os.path.join(res_folder, '{}_exp_aesthetics.json'.format(args.downstream_dataset))))['huggingface_openjourney-v1-0']['classwise'].keys())
        downstream_classes = string_classnames

    # create results paths
    full_results_path = os.path.join(args.results_dir, '{}_{}_{}{}_search_indices.json'.format(args.dataset, args.pt_dataset, args.search_method, args.threshold))
    results_path = os.path.join(args.results_dir, '{}_{}_{}{}_search_counts.json'.format(args.dataset, args.pt_dataset, args.search_method, args.threshold))
    os.makedirs(args.results_dir, exist_ok=True)

    # get all paths of indices
    if args.pt_dataset == 'laion200m_train_test_sim_normalized':
        image_index_paths = [os.path.join(args.features_dir, x) for x in os.listdir(args.features_dir) if '{}_image_search_index_threshold_{}'.format('laion400m', args.threshold) in x and float(x.split('_')[5]) == args.threshold and 'shardstart' in x]
    elif args.pt_dataset == 'laion_aesthetics':
        image_index_paths = [os.path.join(args.features_dir, x) for x in os.listdir(args.features_dir) if '{}_image_search_index_threshold_{}'.format(args.pt_dataset, args.threshold) in x and float(x.split('_')[6]) == args.threshold and 'shardstart' in x]
    else:
        image_index_paths = [os.path.join(args.features_dir, x) for x in os.listdir(args.features_dir) if '{}_image_search_index_threshold_{}'.format(args.pt_dataset, args.threshold) in x and float(x.split('_')[5]) == args.threshold and 'shardstart' in x]

    result_dict = {}

    if args.pt_dataset == 'laion200m_train_test_sim_normalized':
        # matched indices of laion for filtering Mayilvahanan et al's subset
        matching_indices = np.load('../data/laion400m/paths_leave_out_near_val_150m_whole_data_new_pruning_method.npy')

        for index_path in tqdm(image_index_paths, ascii=True, total=len(image_index_paths)):
            # load the inverted index of the pre-training dataset
            image_index = pickle.load(open(index_path, 'rb'))
            # do search
            for d in downstream_classes:
                if d not in image_index.keys():
                    print('{} not in image index'.format(d))
                if d not in result_dict:
                    result_dict[d] = []
                if d in image_index:
                    result_dict[d] += list([int(s.split('.jpg')[0]) for s in image_index[d]])

        # now take set intersections between the mayilvahanan split and the hit-indexes per class
        final_res_dict = {}
        for i in tqdm(result_dict, ascii=True, total=len(result_dict)):
            final_res_dict[i] = [int(x) for x in matching_indices[np.isin(matching_indices, list(set(result_dict[i])))]]
        for i in final_res_dict:
            result_dict[i] = len(final_res_dict[i])

    else:
        for ind_, index_path in enumerate(tqdm(image_index_paths, ascii=True, total=len(image_index_paths))):
            # load the inverted index of the pre-training dataset
            image_index = pickle.load(open(index_path, 'rb'))

            # do search
            for d in downstream_classes:
                if args.pt_dataset != 'laion_aesthetics':
                    if d not in image_index.keys():
                        print('{} not in image index'.format(d))
                        continue
                    else:
                        st = d
                else:
                    st = d
                    if d not in image_index.keys():
                        combs = generate_combinations(d)
                        for cmb in combs:
                            if cmb in image_index:
                                st = cmb
                                break
                    if st not in image_index.keys():
                        print('{} not in image index'.format(st))
                        continue

                if st not in result_dict:
                    result_dict[st] = []

                if args.pt_dataset == 'cc3m':
                    result_dict[st] += list([int(s.split('.jpg')[0]) for s in image_index[st]])
                elif args.pt_dataset == 'laion400m':
                    result_dict[st] += list([int(s.split('.jpg')[0]) for s in image_index[st]])
                elif args.pt_dataset == 'synthci30m':
                    result_dict[st] += list([int(s.split('.jpg')[0].split('/')[1]) for s in image_index[st]])
                else:
                    result_dict[st] += list([int(s) for s in image_index[st]])

        final_res_dict = {}
        for i in tqdm(result_dict, ascii=True, total=len(result_dict)):
            final_res_dict[i] = result_dict[i]
        for i in final_res_dict:
            result_dict[i] = len(final_res_dict[i])

    # dump results
    with open(results_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    with open(full_results_path, 'w') as f:
        json.dump(final_res_dict, f, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dataset", type=str, default='cc3m', choices=['cc3m', 'cc12m', 'yfcc15m', 'laion400m', 'laion_aesthetics', 'laion200m_train_test_sim_normalized'])
    parser.add_argument("--downstream_dataset", type=str, default='cifar10')
    parser.add_argument("--search_method", type=str, default='rampp', choices=['rampp'])
    parser.add_argument("--threshold", type=float, default=0.7, help='threshold used for filtering out rampp outputs')
    parser.add_argument('--cache_dir', type=str, help='cache_dir to save/load CLIP model weights')
    parser.add_argument('--features_dir', type=str, help='dir to save/load CLIP encoded test set features')
    parser.add_argument('--results_dir', type=str, help='dir to save/load results')    
    args = parser.parse_args()
    
    main(args)
