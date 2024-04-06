import os
import numpy as np
from collections import OrderedDict
import argparse
import pandas as pd
from tqdm import tqdm
import re
import pickle
from math import ceil
import gc
import random

def save_as_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def shuffled_indices_func(n):
    indices = list(range(n))
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]
    return indices

def main(args):

    # load the inverted index of the pre-training dataset
    filepath = os.path.join(args.features_dir, '{}_spacy_combined_unigram_dict.pkl'.format(args.pt_dataset))
    with open(filepath, 'rb') as file:
        unigram_index = pickle.load(file)
    print('Loaded index')

    # Calculate the size of each shard
    total_size = len(unigram_index)
    shard_size = ceil(total_size / args.num_shards)

    # Shuffle keys
    keys = list(unigram_index.keys())
    shuffled_indices = shuffled_indices_func(total_size)

    global_key_to_shard_mapping = {}
    print('Permuted samples')
    print('Total size: {}'.format(total_size))

    # Split the data and save each shard
    for i in range(args.num_shards):
        start_index = i * shard_size
        end_index = start_index + shard_size
        shard = {}
        for key_index in shuffled_indices[start_index:end_index]:

            key = keys[key_index]

            shard[key] = unigram_index[key]
            global_key_to_shard_mapping[key] = i

        shard_filepath = "{}_{}.pkl".format(filepath.replace('.pkl', ''), i)
        save_as_pickle(shard, shard_filepath)
        print(f"Saved shard {i} with {len(shard)} items.")

    global_key_to_shard_mapping_file = os.path.join(args.features_dir, '{}_spacy_unigram_to_shard_mapping.pkl'.format(args.pt_dataset))
    save_as_pickle(global_key_to_shard_mapping, global_key_to_shard_mapping_file)
    print("Saved key-to-shard mapping")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dataset", type=str, default='laion400m')
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--max_num_procs", type=int, default=4)
    parser.add_argument('--cache_dir', type=str, help='cache_dir to save/load CLIP model weights')
    parser.add_argument('--features_dir', type=str, help='dir to save/load CLIP encoded test set features')
    parser.add_argument('--results_dir', type=str, help='dir to save/load results')    
    args = parser.parse_args()
    
    main(args)
