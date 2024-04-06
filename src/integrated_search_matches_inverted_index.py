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

def search_for_text(args, unigram_index, multigram_index, search_terms, classname):

    # Retrieve document sets for each term and store in a list
    doc_sets = [set(unigram_index[term]) for term in search_terms if term in unigram_index]

    # Perform set intersection to find common documents
    common_docs = set.intersection(*doc_sets) if doc_sets else set()

    # Optionally handle multigrams here (if applicable)

    return common_docs, classname

def batchify_search_texts(max_num_procs, search_texts, classnames):
    return [search_texts[i:i+max_num_procs] for i in range(0, len(search_texts), max_num_procs)], [classnames[i:i+max_num_procs] for i in range(0, len(search_texts), max_num_procs)]

def main(args):

    ### special case for laion and laion_aesthetics: we already have cached the matching indices for each dataset in:
    ### `<dataset>_laion400m_lemmatized_search_indices.json` inside the results folder
    ### for all other datasets, we do the text search again
    ### TODO: fix this for other datasets too, to cache the indices directly

    ## if not laion, then load the unigram index and do the processing again:
    if args.pt_dataset not in ['laion400m', 'laion_aesthetics']:
        # load the inverted index of the pre-training dataset
        unigram_index = pickle.load(open(os.path.join(args.features_dir, '{}_spacy_combined_unigram_dict.pkl'.format(args.pt_dataset)), 'rb'))
        print('Loaded unigram text index')

    # load multiple image indices as in the image search script
    if args.pt_dataset == 'laion_aesthetics':
        image_index_paths = [os.path.join(args.features_dir, x) for x in os.listdir(args.features_dir) if '{}_image_search_index_threshold_{}'.format(args.pt_dataset, args.image_search_threshold) in x and float(x.split('_')[6]) == float(args.image_search_threshold) and 'shardstart' in x]    
    else:
        image_index_paths = [os.path.join(args.features_dir, x) for x in os.listdir(args.features_dir) if '{}_image_search_index_threshold_{}'.format(args.pt_dataset, args.image_search_threshold) in x and float(x.split('_')[5]) == float(args.image_search_threshold) and 'shardstart' in x]

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


    #### Do text search again if the indices are not cached ###
    if args.pt_dataset not in ['laion400m', 'laion_aesthetics']:
        # do search; split multigrams into unigrams for search and then combine using set intersection
        s_list = []
        print('Processing classname terms for search...')
        processed_classes = [re.split(r'[_\s]+', classname.lower()) for classname in downstream_classes]
        flat_terms = set(term for sublist in processed_classes for term in sublist)
        # other than `lowercased`, all other methods use lemmatization
        if 'lowercased' in args.text_search_method:
            lemmatized_terms = {term: term for term in flat_terms}    
        else:
            lemmatized_terms = {term: token.lemma_ for term in flat_terms for token in nlp(term)}
        search_terms_processed = [[lemmatized_terms[term] for term in class_terms] for class_terms in processed_classes]
        assert len(search_terms_processed) == len(downstream_classes)

        # Batchify search texts
        search_texts_batched, classnames_batched = batchify_search_texts(args.max_num_procs, search_terms_processed, downstream_classes)

        # Using ProcessPoolExecutor to parallelize the search for multiple texts
        print('Running text search')
        text_result_dict = {}
        for batch_id, (batch, curr_classname_batch) in tqdm(enumerate(zip(search_texts_batched, classnames_batched)), ascii=True, total=len(search_texts_batched)):
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(search_for_text, args, unigram_index, multigram_index, search_terms, classn) for search_terms, classn in zip(batch, curr_classname_batch)]
                # Print results as they complete
                for future in futures:
                    hitlist, class_ = future.result()
                    text_result_dict[class_] = hitlist
    elif args.pt_dataset == 'laion400m':
        # do search from cached indices for laion400m
        # get cached indices for text search
        cached_file_path = os.path.join(args.results_dir, '{}_laion400m_{}_search_indices.json'.format(args.downstream_dataset, args.text_search_method))
        with open(cached_file_path) as f:
            text_result_dict = json.load(f)
    elif args.pt_dataset == 'laion_aesthetics':
        # do search from cached indices for laion_aesthetics
        # get cached indices for text search
        cached_file_path = os.path.join(args.results_dir, '{}_laion_aesthetics_{}_search_indices.json'.format(args.downstream_dataset, args.text_search_method))
        with open(cached_file_path) as f:
            text_result_dict = json.load(f)
    else:
        raise ValueError('Unknown pretraining dataset: {}'.format(args.pt_dataset))

    # do image search
    print('Running image search')
    image_result_dict = {}

    if args.pt_dataset == 'laion_aesthetics':
        # do search in cached file path
        cached_file_path = os.path.join(args.results_dir, '{}_laion_aesthetics_rampp{}_search_indices.json'.format(args.downstream_dataset, args.image_search_threshold))
        with open(cached_file_path) as f:
            image_result_dict = json.load(f)
    elif args.pt_dataset == 'laion400m':
        # do search in cached file path
        cached_file_path = os.path.join(args.results_dir, '{}_laion400m_rampp{}_search_indices.json'.format(args.downstream_dataset, args.image_search_threshold))
        with open(cached_file_path) as f:
            image_result_dict = json.load(f)
    else:
        for index_path in tqdm(image_index_paths, ascii=True, total=len(image_index_paths)):
            # load the inverted index of the pre-training dataset
            image_index = pickle.load(open(index_path, 'rb'))
            # do search
            for d in downstream_classes:
                if d not in image_index.keys():
                    print('{} not in image index'.format(d))
                if d not in image_result_dict:
                    image_result_dict[d] = []

                if args.pt_dataset == 'cc3m':
                    image_result_dict[d] += list([int(s.split('.jpg')[0]) for s in image_index[d]])
                elif args.pt_dataset == 'synthci30m':
                    image_result_dict[d] += list([int(s.split('.jpg')[0].split('/')[1]) for s in image_index[d]])
                else:
                    image_result_dict[d] += list(image_index[d])

    # create final results path
    results_path = os.path.join(args.results_dir, '{}_{}_{}_search_counts.json'.format(args.dataset, args.pt_dataset, 'integrated_t{}_i{}'.format(args.text_search_method, args.image_search_threshold)))
    os.makedirs(args.results_dir, exist_ok=True)

    # if pt_dataset is laion_aesthetics, normalize the cub bird names
    if args.pt_dataset == 'laion_aesthetics':
        image_result_dict = {' '.join(k.split('_')).lower():v for k,v in image_result_dict.items()}
        text_result_dict = {' '.join(k.split('_')).lower():v for k,v in text_result_dict.items()}
        ### manually add f1 as 0 to the list to keep same lengths (verified this by hand)
        text_result_dict['f1'] = []

    ### Uncomment for visualisation of results ####
    # print(len(text_result_dict))
    # print(len(image_result_dict))
    # print(sorted(text_result_dict.keys())[:10])
    # print(sorted(image_result_dict.keys())[:10])
    # for k in image_result_dict:
    #     print('image')
    #     print(k, len(image_result_dict[k]))
    #     print(k, sorted(image_result_dict[k])[:20], sorted(image_result_dict[k])[-20:])
    #     print('text')
    #     print(k, len(text_result_dict[k]))
    #     print(k, sorted(text_result_dict[k])[:20], sorted(text_result_dict[k])[-20:])
    #     print('---------')

    # final result
    print('Running merge')
    # sometimes the text result dict misses some keys if there were no hits found, correct for this
    for k in image_result_dict:
        if k not in text_result_dict:
            text_result_dict[k] = []
    assert len(image_result_dict) == len(text_result_dict), 'text and image dicts dont contain same number of concepts'
    integrated_result_dict = {}
    for d in tqdm(downstream_classes, ascii=True, total=len(downstream_classes)):
        integrated_result_dict[d] = len(set(image_result_dict[d]).intersection(set(text_result_dict[d])))
    print(integrated_result_dict)

    # dump results
    with open(results_path, 'w') as f:
        json.dump(integrated_result_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dataset", type=str, default='cc3m')
    parser.add_argument("--downstream_dataset", type=str, default='cifar10')
    parser.add_argument("--text_search_method", type=str, default='lemmatized', choices=['lowercased', 'lemmatized'])
    parser.add_argument("--image_search_threshold", type=str, default='0.7', choices=['0.5', '0.55', '0.6', '0.65', '0.7'])
    parser.add_argument("--max_num_procs", type=int, default=4)
    parser.add_argument('--cache_dir', type=str, help='cache_dir to save/load CLIP model weights')
    parser.add_argument('--features_dir', type=str, help='dir to save/load CLIP encoded test set features')
    parser.add_argument('--results_dir', type=str, help='dir to save/load results')    
    args = parser.parse_args()
    
    main(args)
