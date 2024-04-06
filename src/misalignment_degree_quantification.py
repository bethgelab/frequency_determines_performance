import os
import json
import numpy as np
from collections import OrderedDict
import argparse
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import re
from downstream_dataloader import DataLoader
import pickle
import spacy
nlp = spacy.load("en_core_web_sm")

DATASET_SIZES = {
    'cc3m': 3318333,
    'cc12m': 12423374,
    'yfcc15m': 14825236,
    'laion400m': 414080000, # conservative estimate based on 41.4k shards
    'laion_aesthetics': 161100000, # conservative estimate based on 16.1k shards
}

ALL_DATASETS = [
    'ucf101',
    'cifar10',
    'cifar100',
    'caltech101',
    'caltech256',
    'imagenet',
    'sun397',
    'fgvcaircraft',
    'birdsnap',
    'stanfordcars',
    'cub',
    'flowers102',
    'food101',
    'oxfordpets',
    'dtd',
    'eurosat',
    'imagenet-sketch',
    'imagenet-r',
    'country211',
    'coco',
    'flickr',
]

def save_image_sample2concepts_index(pt_dataset, index_paths, list_of_concepts):


    # iterate over all concepts, create a dict from sample_id -> [], populate the list with the indices of each concept
    if os.path.exists(os.path.join(args.features_dir, '{}_image_index_sample_id_to_concepts.pkl'.format(pt_dataset))):
        return pickle.load(open(os.path.join(args.features_dir, '{}_image_index_sample_id_to_concepts.pkl'.format(pt_dataset)), 'rb'))
    else:

        print('Creating image sample2concept map...')

        if args.pt_dataset in ['laion_aesthetics', 'laion400m']:
            image_result_dict = {}
            if args.pt_dataset == 'laion400m':
                for dset in tqdm(ALL_DATASETS, total=len(ALL_DATASETS), ascii=True):
                    # do search in cached file path
                    cached_file_path = os.path.join(args.results_dir, '{}_{}_rampp{}_search_indices.json'.format(dset, args.pt_dataset, args.image_search_threshold))
                    with open(cached_file_path) as f:
                        image_result_dict.update(json.load(f))
            else:
                for dset in tqdm(['t2i'], total=len(['t2i']), ascii=True):
                    # do search in cached file path
                    cached_file_path = os.path.join(args.results_dir, '{}_{}_rampp{}_search_indices.json'.format(dset, args.pt_dataset, args.image_search_threshold))
                    with open(cached_file_path) as f:
                        image_result_dict.update(json.load(f))
        else:
            # first get image index
            image_result_dict = {}
            for index_path in tqdm(index_paths, ascii=True, total=len(index_paths)):
                # load the inverted index of the pre-training dataset
                image_index = pickle.load(open(index_path, 'rb'))
                # do search
                for d in tqdm(list_of_concepts, ascii=True, total=len(list_of_concepts)):
                    if d not in image_index.keys():
                        print('{} not in image index'.format(d))
                        image_result_dict[d] = []
                        continue
                    if d not in image_result_dict:
                        image_result_dict[d] = []

                    if args.pt_dataset == 'cc3m':
                        image_result_dict[d] += list([int(s.split('.jpg')[0]) for s in image_index[d]])
                    elif args.pt_dataset == 'synthci30m':
                        image_result_dict[d] += list([int(s.split('.jpg')[0].split('/')[1]) for s in image_index[d]])
                    else:
                        image_result_dict[d] += list(image_index[d])

        sample2concept_image_index = {}
        for concept_index, concept in tqdm(enumerate(list_of_concepts), ascii=True, total=len(list_of_concepts)):
            if concept in image_result_dict:
                # for each sample in the current concept's hit list, add it to the sampleid2concepts dict
                for sample_id in image_result_dict[concept]:
                    if sample_id not in sample2concept_image_index:
                        sample2concept_image_index[sample_id] = []
                    sample2concept_image_index[sample_id].append(concept_index)

        for sample_id, sample in enumerate(tqdm(sorted(sample2concept_image_index), ascii=True, total=len(sample2concept_image_index))):
            sample2concept_image_index[sample] = list(set(sample2concept_image_index[sample]))

        ### uncomment this to visualise results ###
        # print('IMAGES')
        # for sample_id, sample in enumerate(sorted(sample2concept_image_index)):
        #     print(sample_id)
        #     print(sample)
        #     print(list(set(sample2concept_image_index[sample])))
        #     print([list_of_concepts[ss] for ss in list(set(sample2concept_image_index[sample]))])
        #     if sample_id == 5:
        #         break

        pickle.dump(sample2concept_image_index, open(os.path.join(args.features_dir, '{}_image_index_sample_id_to_concepts.pkl'.format(pt_dataset)), 'wb'))
        return sample2concept_image_index

def save_text_sample2concepts_index(pt_dataset, index, list_of_concepts):
    # iterate over all concepts, create a dict from sample_id -> [], populate the list with the indices of each concept

    if os.path.exists(os.path.join(args.features_dir, '{}_text_index_sample_id_to_concepts.pkl'.format(pt_dataset))):
        return pickle.load(open(os.path.join(args.features_dir, '{}_text_index_sample_id_to_concepts.pkl'.format(pt_dataset)), 'rb'))
    else:
        print('Creating text sample2concept map...')
        sample2concept_text_index = {}
        for concept_index, concept in tqdm(enumerate(list_of_concepts), total=len(list_of_concepts), ascii=True):
            if concept in index:
                # for each sample in the current concept's hit list, add it to the sampleid2concepts dict
                for sample_id in index[concept]:
                    if sample_id not in sample2concept_text_index:
                        sample2concept_text_index[sample_id] = []
                    sample2concept_text_index[sample_id].append(concept_index)

        for sample_id, sample in enumerate(tqdm(sorted(sample2concept_text_index), ascii=True, total=len(sample2concept_text_index))):
            sample2concept_text_index[sample] = list(set(sample2concept_text_index[sample]))

        ### uncomment this to visualise results ###
        # print('TEXTS')
        # for sample_id, sample in enumerate(sorted(sample2concept_text_index)):
        #     print(sample_id)
        #     print(sample)
        #     print(list(set(sample2concept_text_index[sample])))
        #     print([list_of_concepts[ss] for ss in list(set(sample2concept_text_index[sample]))])
        #     if sample_id == 5:
        #         break

        pickle.dump(sample2concept_text_index, open(os.path.join(args.features_dir, '{}_text_index_sample_id_to_concepts.pkl'.format(pt_dataset)), 'wb'))
        return sample2concept_text_index

def main(args):

    ### special case for laion: we already have cached the matching indices for each dataset in:
    ### `<dataset>_laion400m_lemmatized_search_indices.json` inside the results folder
    ### for all other datasets, we do the text search again
    ### TODO: fix this for other datasets too, to cache the indices directly

    with open('../gpt_descriptions/rampp_overall.json', 'r') as f:
        json_dict = json.load(f)

    list_of_concepts = [list(json_dict[i].keys())[0] for i in range(len(json_dict))]

    # load the inverted index of the pre-training dataset
    if args.pt_dataset not in ['laion_aesthetics', 'laion400m']:
        unigram_index = pickle.load(open(os.path.join(args.features_dir, '{}_spacy_combined_unigram_dict.pkl'.format(args.pt_dataset)), 'rb'))
        print('Loaded unigram text index')
    else:
        unigram_index = {}
        if args.pt_dataset == 'laion400m':
            for dset in tqdm(ALL_DATASETS, total=len(ALL_DATASETS), ascii=True):
                cached_file_path = os.path.join(args.results_dir, '{}_{}_{}_search_indices.json'.format(dset, args.pt_dataset, args.text_search_method))
                with open(cached_file_path) as f:
                    text_result_dict = json.load(f)
                    unigram_index.update(text_result_dict)
        else:
            for dset in tqdm(['t2i'], total=len(['t2i']), ascii=True):
                cached_file_path = os.path.join(args.results_dir, '{}_{}_{}_search_indices.json'.format(dset, args.pt_dataset, args.text_search_method))
                with open(cached_file_path) as f:
                    text_result_dict = json.load(f)
                    unigram_index.update(text_result_dict)
        print('Loaded unigram text index')

    sample2concept_text_index = save_text_sample2concepts_index(args.pt_dataset, unigram_index, list_of_concepts)

    # load multiple image indices as in the image search script
    if args.pt_dataset == 'laion_aesthetics':
        image_index_paths = [os.path.join(args.features_dir, x) for x in os.listdir(args.features_dir) if '{}_image_search_index_threshold_{}'.format(args.pt_dataset, args.image_search_threshold) in x and float(x.split('_')[6]) == float(args.image_search_threshold) and 'shardstart' in x]    
    else:
        image_index_paths = [os.path.join(args.features_dir, x) for x in os.listdir(args.features_dir) if '{}_image_search_index_threshold_{}'.format(args.pt_dataset, args.image_search_threshold) in x and float(x.split('_')[5]) == float(args.image_search_threshold) and 'shardstart' in x]
    sample2concept_image_index = save_image_sample2concepts_index(args.pt_dataset, image_index_paths, list_of_concepts)

    ### uncomment this to visualise results ###
    # print('MAIN IMAGES')
    # for ind in sorted(list(sample2concept_image_index.keys()))[:10]:
    #     print(ind)
    #     print([list_of_concepts[x] for x in sample2concept_image_index[ind]])

    # print('MAIN TEXTS')
    # for ind in sorted(list(sample2concept_text_index.keys()))[:10]:
    #     print(ind)
    #     print([list_of_concepts[x] for x in sample2concept_text_index[ind]])

    # take the intersection of the samples extracted from the imagesample2concept and textsample2concept dicts
    print('Taking set intersection of the samples extracted...')
    intersecting_samples = list(set(list(sample2concept_image_index.keys())).intersection(set(list(sample2concept_text_index.keys()))))

    ### uncomment this to visualise results ###
    # print(intersecting_samples[:10])
    # for indd, s in enumerate(sorted(intersecting_samples)):
    #     print(s)
    #     print([list_of_concepts[x] for x in sample2concept_image_index[s]])
    #     print([list_of_concepts[x] for x in sample2concept_text_index[s]])
    #     if indd == 10:
    #         break

    # get misalignment degree:
    # for each sample, take the intersection of the textsample2concept and imagesample2concept dicts
    # if intersection list is 0, add to misalignment metric else don't
    print('Computing misalignment metrics...')
    misalignment_degree = 0
    shown_misaligned = 0
    for s_index, s in tqdm(enumerate(sorted(intersecting_samples)), ascii=True, total=len(intersecting_samples)):
        if len(list(set(sample2concept_image_index[s]).intersection(set(sample2concept_text_index[s])))) == 0:

            ### uncomment this to visualise results ###
            # if shown_misaligned < 50:
            #     print('Sample index: {}'.format(s))
            #     print('Text concept indexes identified: {}'.format(sample2concept_text_index[s]))
            #     print('Text concepts identified: {}'.format([list_of_concepts[xx] for xx in sample2concept_text_index[s]]))
            #     print('Image concept indexes identified: {}'.format(sample2concept_image_index[s]))
            #     print('Image concepts identified: {}'.format([list_of_concepts[xx] for xx in sample2concept_image_index[s]]))
            #     print('-----------')
            #     shown_misaligned += 1

            misalignment_degree += 1

    print('Absolute misalignment counts: {}'.format(misalignment_degree))
    print('Misalignment degree (percent); with extracted dataset size: {}'.format(misalignment_degree/max(len(sample2concept_image_index), len(sample2concept_text_index))))
    print('Misalignment degree (percent); with fixed conservative dataset size: {}'.format(misalignment_degree/DATASET_SIZES[args.pt_dataset]))

    # dump results
    results_path = os.path.join(args.results_dir, '{}_misalignment_degree.json'.format(args.pt_dataset))
    with open(results_path, 'w') as f:
        json.dump({'misalignment_counts': misalignment_degree, 'misalignment_degree (percent); extracted dataset size': misalignment_degree/max(len(sample2concept_image_index), len(sample2concept_text_index)), 'misalignment_degree (percent); fixed conservative dataset size': misalignment_degree/DATASET_SIZES[args.pt_dataset]}, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dataset", type=str, default='cc3m')
    parser.add_argument("--text_search_method", type=str, default='lemmatized', choices=['lowercased', 'lemmatized'])
    parser.add_argument("--image_search_threshold", type=str, default='0.7', choices=['0.5', '0.55', '0.6', '0.65', '0.7'])
    parser.add_argument("--max_num_procs", type=int, default=4)
    parser.add_argument('--cache_dir', type=str, help='cache_dir to save/load CLIP model weights')
    parser.add_argument('--features_dir', type=str, help='dir to save/load CLIP encoded test set features')
    parser.add_argument('--results_dir', type=str, help='dir to save/load results')    
    args = parser.parse_args()
    
    main(args)
