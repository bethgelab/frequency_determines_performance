import os
import json
import argparse

from concept_extraction import *
from utils import *
from scores_collected import *


def main():
    parser = argparse.ArgumentParser(description="Saving JSON files for t2i from the HEIM website")

    parser.add_argument(
        "--root_dir", default="data/",
        help="root directory to save all results", type=str)

    parser.add_argument(
        "--save_dir", default="results/",
        help="root directory to save all results", type=str)

    parser.add_argument("--dataset",
                        default='cub200',
                        help="Which dataset to use. Refer to the README for more", type=str)

    parser.add_argument('--combine', action='store_true', help='Combine all datasets')
    args = parser.parse_args()

    # List of valid models
    models = ["AlephAlpha_m-vader", "DeepFloyd_IF-I-L-v1.0", "DeepFloyd_IF-I-M-v1.0", "DeepFloyd_IF-I-XL-v1.0",
              "adobe_giga-gan", "craiyon_dalle-mega", "craiyon_dalle-mini", "huggingface_dreamlike-diffusion-v1-0",
              "huggingface_dreamlike-photoreal-v2-0","huggingface_openjourney-v1-0", "huggingface_openjourney-v2-0",
              "huggingface_promptist-stable-diffusion-v1-4","huggingface_redshift-diffusion",
              "huggingface_stable-diffusion-safe-max","huggingface_stable-diffusion-safe-medium",
              "huggingface_stable-diffusion-safe-strong", "huggingface_stable-diffusion-safe-weak",
              "huggingface_stable-diffusion-v1-4", "huggingface_stable-diffusion-v1-5",
              "huggingface_stable-diffusion-v2-1-base","huggingface_stable-diffusion-v2-base",
              "huggingface_vintedois-diffusion-v0-1", "kakaobrain_mindall-e", "lexica_search-stable-diffusion-1.5"]

    if not args.combine:
        for model in models:
            print("Using model: ", model)
            if args.dataset == "mscoco" or args.dataset == "winoground":
                results = winoground_mscoco(model, args)

            elif args.dataset == "cub200":
                results = cub200(model, args)

            else:
                results = other_models(model, args)

            with open(os.path.join(args.root_dir, 'words_uniq.txt'), 'r') as file:
                concepts = [line.rstrip('\n') for line in file]
            concepts = [x.lower() for x in concepts]
            # concepts = list(dict.fromkeys(concepts))
            concepts_all = find_word_indices([sublist[0] for sublist in results], concepts)
            concepts_copy = concepts_all.copy()

            aggregated_scores = calculate_aggregated_scores(concepts_all, results)

            if not os.path.exists(os.path.join(args.save_dir, args.dataset)):
                os.makedirs(os.path.join(args.save_dir, args.dataset))
            with open(os.path.join(args.save_dir, args.dataset, f'{model}_score_all.pkl'), 'wb') as f:
                pickle.dump(aggregated_scores, f)

        if args.dataset == "mscoco" or args.dataset == "winoground":
            winoground_mscoco_collect(args)

        elif args.dataset == "cub200":
            cub200_collect(args)

        else:
            other_models_collect(args)
    else:
        # BRING IT ALL TOGETHER
        all_files = []
        for root, dirs, files in os.walk(args.save_dir):
            for file in files:
                if file != '.DS_Store' and file.endswith('.json'):  # Exclude .DS_Store files
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)

        hae, ha, ma, ea, mc, ec, el, es, ep, eu = [], [], [], [], [], [], [], [], [], []
        task_lists = {'human_aesthetics': hae, 'human_align': ha, 'max_aesthetics': ma,
                      'exp_aesthetics': ea, 'max_clip': mc, 'exp_clip': ec,
                      'exp_lpips': el, 'exp_ssim': es, 'exp_psnr': ep, 'exp_uiqi': eu}
        # categorise the files
        for file in all_files:
            task = '_'.join(file.split('_')[-2:]).split('.')[0]
            if task in task_lists:
                task_lists[task].append(file)

        for task, file_list in task_lists.items():
            print(f"Task: {task}")
            all_concepts = {model_name: {'full': 0.0, 'classwise': {}} for model_name in models}

            for file in file_list:
                with open(file, 'r') as f:
                    # print(file)
                    data = json.load(f)
                    for model, model_data in data.items():
                        # all_concepts[model]['full'].append(model_data['full'])
                        all_concepts[model]['full'] += (model_data['full'] * sum(
                            len(sublist) for sublist in model_data['classwise'].values()))
                        for key, value in data[model]['classwise'].items():
                            if key not in all_concepts[model]['classwise']:
                                all_concepts[model]['classwise'][key] = value
                            else:
                                all_concepts[model]['classwise'][key] += value

            for model, model_data in all_concepts.items():
                sorted_keys = sorted(model_data['classwise'].keys())
                all_concepts[model]['classwise'] = {key: model_data['classwise'][key] for key in sorted_keys}
                model_data['full'] = model_data['full'] / sum(
                    len(sublist) for sublist in model_data['classwise'].values())

            with open(os.path.join(args.save_dir, f'{task}.json'), 'w') as f:
                json.dump(all_concepts, f, indent=4)

if __name__ == '__main__':
    main()