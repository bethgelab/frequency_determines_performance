import os
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
import gc

def main(args):

    output_pickles = sorted([os.path.join(args.results_dir, 'rampp_outputs', args.pt_dataset, x) for x in os.listdir(os.path.join(args.results_dir, 'rampp_outputs', args.pt_dataset)) if '.pkl' in x])
    o_ = pickle.load(open(os.path.join(args.features_dir, 'rampp_categories.pkl'), 'rb'))
    tag_list = np.asarray(o_[1])
    image_search_index = {}

    for threshold in args.thresholds:
        image_search_index[threshold] = {t:[] for t in tag_list}

    for index, pickle_file in tqdm(enumerate(output_pickles), ascii=True, total=len(output_pickles)):

        if index < args.start_index_id:
            continue

        if index >= args.end_index_id:
            continue

        if args.pt_dataset == 'laion_aesthetics':
            curr_pkl_file_index = int(pickle_file.split('/')[-1].split('_')[-1].split('.pkl')[0])

        if args.pt_dataset == 'cc12m':
            curr_shardid_to_csvid_mapping = pickle.load(open(os.path.join('../cc12m/mappings_from_shard_ids_to_csv_ids', 'cc12m_shard_ids_to_csv_id_mapping_{}.pkl'.format(index)), 'rb'))

        dump = pickle.load(open(pickle_file, 'rb'))
        filenames = dump['filenames']
        probs = torch.tensor(np.asarray(dump['probs'])).cpu()

        for threshold in args.thresholds:

            targets = torch.where(
                probs > torch.tensor(threshold).cpu(),
                torch.tensor(1.0).cpu(),
                torch.zeros(len(tag_list)).cpu())

            for index_, (f, p, t) in enumerate(zip(filenames, probs.numpy(), targets.numpy())):
                curr_tags = tag_list[np.argwhere(t==1)].squeeze(axis=1)
                for tag in curr_tags:
                    if args.pt_dataset == 'cc12m':
                        image_search_index[threshold][tag].append(curr_shardid_to_csvid_mapping[f])    
                    elif args.pt_dataset == 'laion_aesthetics':
                        image_search_index[threshold][tag].append(int(f.split('.jpg')[0]) + curr_pkl_file_index * 10000)
                    else:
                        image_search_index[threshold][tag].append(f)

    del dump
    gc.collect()

    for threshold in args.thresholds:
        with open(os.path.join(args.features_dir, '{}_image_search_index_threshold_{}_shardstart{}_shardend{}.pkl'.format(args.pt_dataset, threshold, args.start_index_id, args.end_index_id)), 'wb') as f:
            pickle.dump(image_search_index[threshold], f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dataset", type=str, default='laion400m')
    parser.add_argument("--start_index_id", type=int, default=0)
    parser.add_argument("--end_index_id", type=int, default=2000)
    parser.add_argument("--thresholds", nargs='+', type=float, default=[0.7], help='threshold for filtering the rampp predictions over all classes')
    parser.add_argument('--cache_dir', type=str, help='cache_dir to save/load CLIP model weights')
    parser.add_argument('--features_dir', type=str, help='dir to save/load CLIP encoded test set features')
    parser.add_argument('--results_dir', type=str, help='dir to save/load results')    
    args = parser.parse_args()
    
    main(args)