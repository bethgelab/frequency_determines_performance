import json
import spacy
import os
import pickle
from nltk.corpus import wordnet as wn


def score_index(score, metric):
    index = None
    for i, entry in enumerate(data_score):
        for stat in entry['stats']:
            if stat['name']['name'] == metric:
                index = i
                break
        if index is not None:
            break

    return index


def find_word_indices(prompts, words):
    word_indices = {}

    for word in words:
        index_list = []
        for index, prompt in enumerate(prompts):
            if f' {word} ' in f' {prompt} ':  # Ensure exact word match
                index_list.append(index)
        word_indices[word] = index_list
    word_indices_final = {key: value for key, value in word_indices.items() if value != []}
    return word_indices_final


def concept_mod(dict, size):
    new_dict = {}
    for key, value in dict.items():
        if len(value) >= size:
            new_dict[key] = value
    return new_dict


def calculate_average_scores(indices, scores):
    average_scores = {}

    for key, index_list in indices.items():
        key_scores = []
        for index in index_list:
            if index < len(scores):
                key_scores.append(scores[index][1:])  # Skipping index 0 and taking every other element

        if key_scores:
            num_sublists = len(key_scores)
            sublist_length = len(key_scores[0])
            averaged_sublist = [sum(col) / num_sublists for col in zip(*key_scores)]
            average_scores[key] = averaged_sublist
        else:
            average_scores[key] = None  # for empty index lists

    return average_scores


# get root path
dataset = "draw_bench"

#root for the per-model score and prompt files. For me it is in t2i/data/. Change this to your root directory
root_dir = "data/"

save_dir = "results/"
all_prompts = []
models = [f for f in os.listdir(root_dir + dataset) if os.path.isdir(os.path.join(root_dir, dataset, f))]
for model in models:
    print("Using model: ", model)
    score_dir = os.path.join(root_dir, dataset, model, "score/")
    prompt_dir = os.path.join(root_dir, dataset, model, "prompt/")

    categories = [f for f in sorted(os.listdir(score_dir)) if not f.startswith('.')]

    results = []
    for category in categories:
        print("Using category: ", category)
        with open(os.path.join(score_dir, category)) as f:
            data_score = json.load(f)

        with open(os.path.join(prompt_dir, category)) as f:
            data_prompt = json.load(f)

        for i in range(len(data_prompt)):
            if dataset == 'coco':
                #have to do this because COCO does not have the same number of scores for each prompt
                # therefore there are inconsistent indices
                aesthetic_index = score_index(data_score, "expected_aesthetics_score")
                item_aesthetic = data_score[i + aesthetic_index]

                clip_index = score_index(data_score, 'expected_clip_score')
                item_clip = data_score[i + clip_index]

                human_index = score_index(data_score, 'image_text_alignment_human')
                item_human = data_score[i + human_index]
            else:
                item_aesthetic = data_score[i]
                item_clip = data_score[i + len(data_prompt)]

            instance_id = data_prompt[i]['id']
            aesthetic_id = item_aesthetic['instance_id']
            clip_id = item_clip['instance_id']
            if dataset == 'coco':
                human_id = item_human['instance_id']
                assert aesthetic_id == clip_id == instance_id == human_id
            else:
                assert aesthetic_id == clip_id == instance_id

            exp_aesthetic_score_sum = item_aesthetic['stats'][0]['sum']
            max_aesthetic_score_sum = item_aesthetic['stats'][1]['sum']
            exp_clip_score_sum = item_clip['stats'][0]['sum']
            max_clip_score_sum = item_clip['stats'][1]['sum']

            if dataset == 'coco':
                mean_align = item_human['stats'][0]['mean']
                mean_aesthetic = item_human['stats'][2]['mean']
                results.append([data_prompt[i]['input']['text'].lower(),
                                exp_aesthetic_score_sum, max_aesthetic_score_sum,
                                exp_clip_score_sum, max_clip_score_sum,
                                mean_align, mean_aesthetic])
            else:
                results.append([data_prompt[i]['input']['text'].lower(),
                                exp_aesthetic_score_sum, max_aesthetic_score_sum,
                                exp_clip_score_sum, max_clip_score_sum])

    #load the concepts from the unigram dict
    with open(f'/concepts/{dataset}/test_0_unigram_dict.pkl', 'rb') as f:
        concepts = pickle.load(f)

    concepts = list(concepts.keys())
    concepts = [x.lower() for x in concepts]
    concepts = list(dict.fromkeys(concepts))

    #get the indices of the prompts where the concept is found
    concepts_all = find_word_indices([sublist[0] for sublist in results], concepts)
    # concepts_copy = concepts_all.copy()
    # for concept in concepts_copy:
    #     syn = wn.synsets(concept)
    #     if not syn:
    #         continue
    #     else:
    #         tmp = syn[0].pos()
    #         if tmp == 'v' or tmp == 'a' or tmp == 's':
    #             print(concept)
    #             print(syn[0].pos())
    #             del concepts_all[concept]
    # if dataset == 'parti':
    #     rem = ['orange', 'yellow', 'red', 'blue', 'green', 'times', 'great', 'inside', 'style', 'art', 'left',
    #            'right', '', 'a', 'smiling', 'liberty', 'one', 'white']
    # elif dataset == 'draw_bench':
    #     rem = ['or', 'right', 'left', 'green', 'middle', 'view', 'side', 'it', 'orange']
    # else:
    #     rem = ['their', 'his', 'talking', 'an', 'the', 'to', 'are', 'two', 'large', 'standing', 'looking', 'black',
    #            'white', 'blue', 'serving', 'it', 'at', 'along', 'waiting', 'riding', 'walking', 'in', 'old', 'young',
    #            'small', 'surfing', 'taking', 'down', 'a', 'holding', 'wearing', 'into', 'three', 'this', 'little',
    #            'living', 'green', 'red', 'sitting']
    #
    # for re in rem:
    # del concepts_all[re]
    #
    # if dataset == 'draw_bench' or dataset == 'coco':
    #     thres = 2
    # elif dataset == 'parti':
    #     thres = 5
    #
    # concepts_thres = concept_mod(concepts_all, thres)

    avg_scores_all = calculate_average_scores(concepts_all, results)
    # avg_scores_thres = calculate_average_scores(concepts_thres, results)

    if not os.path.exists(os.path.join(save_dir, dataset)):
        os.makedirs(os.path.join(save_dir, dataset))
    with open(os.path.join(save_dir, dataset, f'{model}_score_all.pkl'), 'wb') as f:
       pickle.dump(avg_scores_all, f)

    # with open(os.path.join(save_dir, dataset, f'{model}_score_{thres}.pkl'), 'wb') as f:
    # pickle.dump(avg_scores_thres, f)
    # print("Number of results",len(results))