import os
import json
import argparse
import pickle

from utils import *
def winoground_mscoco(model, args):
    if args.dataset == "mscoco":
        score_file = os.path.join(args.root_dir, args.dataset, model, "score/mscoco_base.json")
        prompt_file = os.path.join(args.root_dir, args.dataset, model, "prompt/mscoco_base.json")
    else:
        score_file = os.path.join(args.root_dir, args.dataset, model, "score/scoreout.json")
        prompt_file = os.path.join(args.root_dir, args.dataset, model, "prompt/promptout.json")

    with open(os.path.join(score_file)) as f:
        data_score = json.load(f)
    with open(os.path.join(prompt_file)) as f:
        data_prompt = json.load(f)
    results = []

    init_aesthetics, fin_aesthetics = score_index(data_score, 'expected_aesthetics_score')
    init_clip, fin_clip = score_index(data_score, 'expected_clip_score')
    init_human, fin_human = score_index(data_score, 'image_text_alignment_human')
    init_psnr, fin_psnr = score_index(data_score, 'expected_psnr_score')
    init_lpips, fin_lpips = score_index(data_score, 'expected_lpips_score')
    init_uiqi, fin_uiqi = score_index(data_score, 'expected_uiqi_score')
    init_ssim, fin_ssim = score_index(data_score, 'expected_multi_scale_ssim_score')

    aesthetics_data = data_score[init_aesthetics: fin_aesthetics]
    clip_data = data_score[init_clip: fin_clip]
    human_data = data_score[init_human: fin_human]
    psnr_data = data_score[init_psnr:fin_psnr]
    lpips_data = data_score[init_lpips:fin_lpips]
    uiqi_data = data_score[init_uiqi: fin_uiqi]
    ssim_data = data_score[init_ssim:fin_ssim]

    for i in range(len(data_prompt)):
        id = data_prompt[i]['id']

        item_aesthetics = aesthetics_data[i]
        item_clip = clip_data[i]
        item_human = find_entry_by_instance_id(human_data, id)
        item_psnr = find_entry_by_instance_id(psnr_data, id)
        item_lpips = find_entry_by_instance_id(lpips_data, id)
        item_uiqi = find_entry_by_instance_id(uiqi_data, id)
        item_ssim = find_entry_by_instance_id(ssim_data, id)

        assert id == item_aesthetics['instance_id'] == item_clip['instance_id'] == item_human['instance_id'] == \
               item_psnr['instance_id'] \
               == item_lpips['instance_id'] == item_uiqi['instance_id'] == item_ssim['instance_id']

        exp_aesthetics_score_sum = item_aesthetics['stats'][0]['sum']
        max_aesthetics_score_sum = item_aesthetics['stats'][1]['sum']
        exp_clip_score_sum = item_clip['stats'][0]['sum']
        max_clip_score_sum = item_clip['stats'][1]['sum']
        if item_human['stats'] is None:
            exp_human_align = 'skip'
            exp_human_aesthetics = 'skip'
        else:
            exp_human_align = item_human['stats'][0]['mean']
            if len(item_human['stats']) == 1:
                exp_human_aesthetics = 'skip'
            else:
                exp_human_aesthetics = item_human['stats'][2]['mean']

        if item_uiqi['stats'] is None:
            exp_uiqi_score_sum = 'skip'
        else:
            exp_uiqi_score_sum = item_uiqi['stats'][0]['sum']

        exp_lpips_score_sum = item_lpips['stats'][0]['sum']
        exp_ssim_score_sum = item_ssim['stats'][0]['sum']
        exp_psnr_score_sum = item_psnr['stats'][0]['sum']

        results.append([data_prompt[i]['input']['text'].lower(),
                        exp_aesthetics_score_sum, max_aesthetics_score_sum,
                        exp_clip_score_sum, max_clip_score_sum,
                        exp_human_align, exp_human_aesthetics,
                        exp_lpips_score_sum, exp_ssim_score_sum,
                        exp_psnr_score_sum, exp_uiqi_score_sum])
    return results

def cub200(model,args):
    score_file = os.path.join(args.root_dir, args.dataset, model, "score/scoreout.json")
    prompt_file = os.path.join(args.root_dir, args.dataset, model, "prompt/promptout.json")

    with open(os.path.join(score_file)) as f:
        data_score = json.load(f)
    with open(os.path.join(prompt_file)) as f:
        data_prompt = json.load(f)

    init_index_uiqi, final_index_uiqi = score_index(data_score, 'expected_uiqi_score')
    init_index_aesthetics, final_index_aesthetics = score_index(data_score, 'expected_aesthetics_score')
    init_index_clip, final_index_clip = score_index(data_score, 'expected_clip_score')
    uiqi_data = data_score[init_index_uiqi:final_index_uiqi]

    results = []
    for i in range(len(data_prompt)):
        id = data_prompt[i]['id']

        item_lpips = data_score[i]
        item_ssim = data_score[i + len(data_prompt)]
        item_psnr = data_score[i + 2 * len(data_prompt)]

        item_uiqi = find_entry_by_instance_id(uiqi_data, id)
        item_aesthetics = data_score[i + init_index_aesthetics]
        item_clip = data_score[i + init_index_clip]

        assert id == item_lpips['instance_id'] == item_ssim['instance_id'] == item_psnr['instance_id'] == item_uiqi[
            'instance_id'] == item_aesthetics['instance_id'] == item_clip['instance_id']

        exp_lpips_score_sum = item_lpips['stats'][0]['sum']
        exp_ssim_score_sum = item_ssim['stats'][0]['sum']
        exp_psnr_score_sum = item_psnr['stats'][0]['sum']
        if item_uiqi['stats'] is None:
            exp_uiqi_score_sum = 'skip'
        else:
            exp_uiqi_score_sum = item_uiqi['stats'][0]['sum']
        exp_aesthetics_score_sum = item_aesthetics['stats'][0]['sum']
        max_aesthetics_score_sum = item_aesthetics['stats'][1]['sum']
        exp_clip_score_sum = item_clip['stats'][0]['sum']
        max_clip_score_sum = item_clip['stats'][1]['sum']

        file_path = data_prompt[i]['references'][0]['output']['file_path']
        concept = file_path.split('/')[-2].split('.')[1]
        concept = concept.lower().replace('_', ' ')

        results.append([concept,
                        exp_aesthetics_score_sum, max_aesthetics_score_sum,
                        exp_clip_score_sum, max_clip_score_sum,
                        exp_lpips_score_sum, exp_ssim_score_sum,
                        exp_psnr_score_sum, exp_uiqi_score_sum])
    return results

def other_models(model, args):
    score_dir = os.path.join(args.root_dir, args.dataset, model, "score/")
    prompt_dir = os.path.join(args.root_dir, args.dataset, model, "prompt/")
    categories = [f for f in sorted(os.listdir(score_dir)) if not f.startswith('.')]

    results = []
    for category in categories:
        if args.dataset == 'parti_prompts' or args.dataset == 'draw_bench' or args.dataset == 'detection':
            print("Using category: ", category)
            with open(os.path.join(score_dir, category)) as f:
                data_score = json.load(f)

            with open(os.path.join(prompt_dir, category)) as f:
                data_prompt = json.load(f)
        else:
            score_file = os.path.join(args.root_dir, args.dataset, model, "score/scoreout.json")
            prompt_file = os.path.join(args.root_dir, args.dataset, model, "prompt/promptout.json")

            #Categories loop runs once, so this is fine
            with open(os.path.join(score_file)) as f:
                data_score = json.load(f)

            with open(os.path.join(prompt_file)) as f:
                data_prompt = json.load(f)

        init_aesthetics, fin_aesthetics = score_index(data_score, 'expected_aesthetics_score')
        init_clip, fin_clip = score_index(data_score, 'expected_clip_score')
        init_human, fin_human = score_index(data_score, 'image_text_alignment_human')
        aesthetics_data = data_score[init_aesthetics: fin_aesthetics]
        clip_data = data_score[init_clip: fin_clip]
        human_data = data_score[init_human: fin_human]

        for i in range(len(data_prompt)):
            id = data_prompt[i]['id']

            item_aesthetics = aesthetics_data[i]
            item_clip = clip_data[i]
            item_human = find_entry_by_instance_id(human_data, id)

            assert id == item_aesthetics['instance_id'] == item_clip['instance_id'] == item_human['instance_id']

            exp_aesthetics_score_sum = item_aesthetics['stats'][0]['sum']
            max_aesthetics_score_sum = item_aesthetics['stats'][1]['sum']
            exp_clip_score_sum = item_clip['stats'][0]['sum']
            max_clip_score_sum = item_clip['stats'][1]['sum']
            if item_human['stats'] is None:
                exp_human_align = 'skip'
                exp_human_aesthetics = 'skip'
            else:
                exp_human_align = item_human['stats'][0]['mean']
                if len(item_human['stats']) == 1:
                    exp_human_aesthetics = 'skip'
                else:
                    if args.dataset == "daily_dalle":
                        exp_human_aesthetics = item_human['stats'][3]['mean']
                    else:
                        exp_human_aesthetics = item_human['stats'][2]['mean']

            results.append([data_prompt[i]['input']['text'].lower(),
                            exp_aesthetics_score_sum, max_aesthetics_score_sum,
                            exp_clip_score_sum, max_clip_score_sum,
                            exp_human_align, exp_human_aesthetics])
    return results