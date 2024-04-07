import torch
import gc
import numpy as np
import random
import re

def score_index(score, metric):
    initial_index = None
    final_index = None

    for i, entry in enumerate(score):
        for j, stat in enumerate(entry['stats']):
            if stat['name']['name'] == metric:
                if initial_index is None:
                    initial_index = i
                final_index = i  # Update final index as we iterate
                break

    return initial_index, final_index + 1

def find_entry_by_instance_id(data, query_instance_id):
    for entry in data:
        if entry['instance_id'] == query_instance_id:
            return entry
    return {'instance_id': query_instance_id,
            'stats': None}  # Return None if the instance_id is not found


def find_word_indices(prompts, words):
    count = 0
    word_indices = {}

    for word in words:
        index_list = []
        for index, prompt in enumerate(prompts):
            prompt = re.sub(r'[^\w\s]', '', prompt)  # Remove punctuation

            if f' {word} ' in f' {prompt} ' or f' {word}s' in f' {prompt} ' or f' {word}.' in f' {prompt} ' \
                    or f' {word}\'s' in f' {prompt} ' or f' \'{word}\'' in f' {prompt} ' or f' \"{word}\"' in f' {prompt} ' \
                    or f' {word}-' in f' {prompt} ' or f' {word}:' in f' {prompt} ':
                # Ensure exact word match. Added more conditions to handle different cases, also to double-check everything
                index_list.append(index)
                count = count + 1
        word_indices[word] = index_list
    word_indices_final = {key: value for key, value in word_indices.items() if value != []}
    return word_indices_final


def find_words_indices(prompts, words):
    word_indices = {}
    count = 0
    for word in words:
        index_list = []
        for index, prompt in enumerate(prompts):
            if word == prompt:  # Ensure exact word match
                index_list.append(index)
                count = count + 1
        word_indices[word] = index_list
    word_indices_final = {key: value for key, value in word_indices.items() if value != []}
    return word_indices_final


def concept_mod(dict, size):
    new_dict = {}
    for key, value in dict.items():
        if len(value) >= size:
            new_dict[key] = value
    return new_dict


def calculate_aggregated_scores(indices, scores):
    agg_scores = {}

    for key, index_list in indices.items():
        key_scores = []
        agg_scores[key] = []
        for index in index_list:
            if index < len(scores):
                key_scores.append(scores[index][1:])  # Skipping index 0 and taking every other element

        if key_scores:
            agg_scores[key].append(key_scores)
        else:
            agg_scores[key] = None  # for empty index lists
    return agg_scores

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)