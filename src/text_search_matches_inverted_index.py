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

def search_for_text(args, unigram_index, multigram_index, search_terms, classname, return_indices=False):

    # Retrieve document sets for each term and store in a list
    doc_sets = [set(unigram_index[term]) for term in search_terms if term in unigram_index]

    # Perform set intersection to find common documents
    common_docs = set.intersection(*doc_sets) if doc_sets else set()

    # Optionally handle multigrams here (if applicable)

    if return_indices:
        return common_docs, len(common_docs), classname
    return len(common_docs), classname

def batchify_search_texts(max_num_procs, search_texts, classnames):
    return [search_texts[i:i+max_num_procs] for i in range(0, len(search_texts), max_num_procs)], [classnames[i:i+max_num_procs] for i in range(0, len(search_texts), max_num_procs)]

def search_in_full_index(args):

    # load the inverted index of the pre-training dataset
    unigram_index = pickle.load(open(os.path.join(args.features_dir, '{}_spacy_combined_unigram_dict.pkl'.format(args.pt_dataset)), 'rb'))
    print('Loaded unigram index')

    # dummy parameters for dataloader
    args.val_batch_size = 64
    args.train_batch_size = 256
    args.k_shot = 1
    args.dataset = args.downstream_dataset

    # get downstream classes
    if args.dataset not in ['coco', 'flickr']:
        data_loader = DataLoader(args, None)
        _, _, _, _, num_classes, string_classnames = data_loader.load_dataset()
        if args.downstream_dataset == 'cifar10':
            string_classnames[0] = 'airplane'
        downstream_classes = string_classnames
    else:
        downstream_classes = pickle.load(open(os.path.join(args.features_dir, '{}_concept_list.pkl'.format(args.dataset)), 'rb'))

    # do search; split multigrams into unigrams for search and then combine using set intersection
    s_list = []
    print('Processing classname terms for search...')
    processed_classes = [re.split(r'[_\s]+', classname.lower()) for classname in downstream_classes]
    flat_terms = set(term for sublist in processed_classes for term in sublist)
    # other than `lowercased`, all other methods use lemmatization
    if 'lowercased' in args.search_method:
        lemmatized_terms = {term: term for term in flat_terms}    
    else:
        lemmatized_terms = {term: token.lemma_ for term in flat_terms for token in nlp(term)}
    search_terms_processed = [[lemmatized_terms[term] for term in class_terms] for class_terms in processed_classes]
    assert len(search_terms_processed) == len(downstream_classes)

    # Batchify search texts
    search_texts_batched, classnames_batched = batchify_search_texts(args.max_num_procs, search_terms_processed, downstream_classes)

    results_path = os.path.join(args.results_dir, '{}_{}_{}_search_counts.json'.format(args.dataset, args.pt_dataset, args.search_method))

    os.makedirs(args.results_dir, exist_ok=True)

    # Using ProcessPoolExecutor to parallelize the search for multiple texts
    result_dict = {}
    for batch_id, (batch, curr_classname_batch) in tqdm(enumerate(zip(search_texts_batched, classnames_batched)), ascii=True, total=len(search_texts_batched)):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(search_for_text, args, unigram_index, multigram_index, search_terms, classn) for search_terms, classn in zip(batch, curr_classname_batch)]
            # Print results as they complete
            for future in futures:
                count, class_ = future.result()
                result_dict[class_] = count
        # dump results
        with open(results_path, 'w') as f:
            json.dump(result_dict, f, indent=4)

def search_in_chunked_shards(args):

    # load the mapping file from concepts to shard indices
    concept_to_index_shard_id = pickle.load(open(os.path.join(args.features_dir, '{}_spacy_unigram_to_shard_mapping.pkl'.format(args.pt_dataset)), 'rb'))

    # dummy parameters for dataloader
    args.val_batch_size = 64
    args.train_batch_size = 256
    args.k_shot = 1
    args.dataset = args.downstream_dataset

    # get downstream classes
    if args.dataset not in ['coco', 'flickr', 't2idrawbench', 't2iparti', 't2icoco', 't2i']:
        res_folder = '../zero_shot_evaluations'
        string_classnames = list(json.load(open(os.path.join(res_folder, '{}_class_results.json'.format(args.downstream_dataset))))['backboneViT-B-16_pretrainedcc12m']['classwise'].keys())
        if args.downstream_dataset == 'cifar10':
            string_classnames[0] = 'airplane'
        downstream_classes = string_classnames
        if args.downstream_dataset == 'imagenet-r':
            downstream_classes = [d.replace('_', ' ') for d in downstream_classes]
    elif args.dataset in ['coco', 'flickr']:
        res_folder = '../retrieval_evaluations'
        string_classnames = list(json.load(open(os.path.join(res_folder, '{}_i2t_k=1_results.json'.format(args.downstream_dataset))))['backboneViT-B-16_pretrainedcc12m']['conceptwise'].keys())
        downstream_classes = string_classnames
        if args.downstream_dataset == 'coco':
            downstream_classes.remove('')
    elif args.dataset in ['t2icoco', 't2idrawbench', 't2iparti', 't2i']:
        res_folder = '../t2i_evaluations'
        string_classnames = list(json.load(open(os.path.join(res_folder, '{}_exp_aesthetics.json'.format(args.downstream_dataset))))['huggingface_openjourney-v1-0']['classwise'].keys())
        downstream_classes = string_classnames

    # map concepts to their corresponding shards
    rel_shard_ids = np.unique(list(concept_to_index_shard_id.values()))
    shard_concept_maps_ = {i:[] for i in rel_shard_ids}
    classname_to_unigrams_map = {}
    for dc in downstream_classes:
        class_unigrams = re.split(r'[_\s]+', dc.lower())
        classname_to_unigrams_map[dc] = []
        for cu in class_unigrams:
            # NOTE: we are excluding all concepts that do not occur in the map
            # this is to ensure we don't get 0 hits artificially and increase recall
            # e.g., kite (bird of prey) will be mapped --> kite , (bird , of, prey)
            # we remove (bird and prey) to ensure that we don't count 0s for these tokens
            try:
                shard_concept_maps_[concept_to_index_shard_id[cu]].append(cu)
                classname_to_unigrams_map[dc].append(cu)
            except KeyError as e:
                pass
    shard_concept_maps = {k:v for k,v in shard_concept_maps_.items() if len(v)>0}

    # create output paths
    results_path = os.path.join(args.results_dir, '{}_{}_{}_search_counts.json'.format(args.dataset, args.pt_dataset, args.search_method))
    full_results_path = os.path.join(args.results_dir, '{}_{}_{}_search_indices.json'.format(args.dataset, args.pt_dataset, args.search_method))
    os.makedirs(args.results_dir, exist_ok=True)

    full_result_dict = {}

    for shard_index, subset_classes in tqdm(shard_concept_maps.items(), ascii=True, total=len(shard_concept_maps)):

        # load the inverted index of the pre-training dataset
        unigram_index = pickle.load(open(os.path.join(args.features_dir, '{}_spacy_combined_unigram_dict_{}.pkl'.format(args.pt_dataset, shard_index)), 'rb'))
        print('Loaded unigram index')

        # do search; split multigrams into unigrams for search and then combine using set intersection
        s_list = []
        print('Processing classname terms for search...')
        processed_classes = [re.split(r'[_\s]+', classname.lower()) for classname in subset_classes]
        flat_terms = set(term for sublist in processed_classes for term in sublist)
        # other than `lowercased`, all other methods use lemmatization
        if 'lowercased' in args.search_method:
            lemmatized_terms = {term: term for term in flat_terms}    
        else:
            lemmatized_terms = {term: token.lemma_ for term in flat_terms for token in nlp(term)}
        search_terms_processed = [[lemmatized_terms[term] for term in class_terms] for class_terms in processed_classes]
        assert len(search_terms_processed) == len(subset_classes)

        # Batchify search texts
        search_texts_batched, classnames_batched = batchify_search_texts(args.max_num_procs, search_terms_processed, subset_classes)

        # Using ProcessPoolExecutor to parallelize the search for multiple texts
        for batch_id, (batch, curr_classname_batch) in tqdm(enumerate(zip(search_texts_batched, classnames_batched)), ascii=True, total=len(search_texts_batched)):
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(search_for_text, args, unigram_index, multigram_index, search_terms, classn, True) for search_terms, classn in zip(batch, curr_classname_batch)]
                # Print results as they complete
                for future in futures:
                    matched_indices, count, class_ = future.result()
                    full_result_dict[class_] = matched_indices

    # merge results for all classes
    output_res_dict = {}
    full_output_res_dict = {}
    for cn in classname_to_unigrams_map:
        if len(classname_to_unigrams_map[cn]) > 0:
            r = full_result_dict[classname_to_unigrams_map[cn][0]]
            for ind in range(1, len(classname_to_unigrams_map[cn])):
                r = set.intersection(*[r, full_result_dict[classname_to_unigrams_map[cn][ind]]])
            output_res_dict[cn] = len(r)
            full_output_res_dict[cn] = [int(x) for x in list(r)]
        else:
            output_res_dict[cn] = 0
            full_output_res_dict[cn] = []

    # dump results
    with open(results_path, 'w') as f:
        json.dump(output_res_dict, f, indent=4)
    with open(full_results_path, 'w') as f:
        json.dump(full_output_res_dict, f, indent=4)

def main(args):
    if args.do_chunked_search:
        search_in_chunked_shards(args)
    else:
        search_in_full_index(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dataset", type=str, default='cc3m')
    parser.add_argument("--downstream_dataset", type=str, default='cifar10')
    # Different search method descriptions:
    # 1. lowercased: plain text-only search with only lowercasing as default preprocessing step on queries
    # 2. lemmatized: text-only search with lowercasing and lemmatization as preprocessing step on queries
    parser.add_argument("--search_method", type=str, default='lemmatized', choices=['lowercased', 'lemmatized'])
    parser.add_argument("--do_chunked_search", type=bool, default=False, help="whether to search in chunked index shards or in the full unsharded index")
    parser.add_argument("--max_num_procs", type=int, default=4)
    parser.add_argument('--cache_dir', type=str, help='cache_dir to save/load CLIP model weights')
    parser.add_argument('--features_dir', type=str, help='dir to save/load CLIP encoded test set features')
    parser.add_argument('--results_dir', type=str, help='dir to save/load results')    
    args = parser.parse_args()
    
    main(args)
