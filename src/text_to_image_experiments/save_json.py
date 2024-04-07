import requests
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Saving JSON files for t2i from the HEIM website")
    parser.add_argument(
        "--save_root", default="data/",
        help="root directory to save json files", type=str)

    parser.add_argument("--dataset",
                        default='cub200',
                        help="Which dataset to use. Refer to the README for more", type=str)

    args = parser.parse_args()

    if args.dataset == "parti_prompts":
        categories = ["Animals", "Artifacts", "Arts", "Food", "Illustrations", "Indoor", "Outdoor", "People", "Produce",
                      "Vehicles", "World"]

    elif args.dataset == "draw_bench":
        categories = ["Colors", "Conflicting", "Counting", "DALL-E", "Descriptions", "Gary", "Positional", "Reddit", "Text"]

    elif args.dataset == "mscoco":
        categories = ["mscoco_base"]

    elif args.dataset == "detection":
        categories = ["object"]
    else:
        categories = [""]  # No category required in the url

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

    save_dir = os.path.join(args.save_root, args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for category in categories:
        for model in models:
            prompt_dir = os.path.join(save_dir, model, 'prompt')
            score_dir = os.path.join(save_dir, model, 'score')
            [os.makedirs(directory, exist_ok=True) for directory in [prompt_dir, score_dir]]

            if category == "":
                print(f"Saving prompts and score for {args.dataset} and {model}")
                prompt_file = os.path.join(prompt_dir, 'promptout.json')
                score_file = os.path.join(score_dir, 'scoreout.json')

                prompt_url = f"https://storage.googleapis.com/crfm-helm-public/heim/benchmark_output/runs/v1.1.0/{args.dataset}:model={model},max_eval_instances=100/instances.json"
                score_url = f"https://storage.googleapis.com/crfm-helm-public/heim/benchmark_output/runs/v1.1.0/{args.dataset}:model={model},max_eval_instances=100/per_instance_stats.json"
            else:
                print(f"Saving prompts and score for {category} in {args.dataset} and {model}")
                prompt_file = os.path.join(prompt_dir, category.lower() + '.json')
                score_file = os.path.join(score_dir, category.lower() + '.json')

                if args.dataset == 'detection':
                    prompt_url = f"https://storage.googleapis.com/crfm-helm-public/heim/benchmark_output/runs/v1.1.0/{args.dataset}:skill={category},model={model},max_eval_instances=100/instances.json"
                    score_url = f"https://storage.googleapis.com/crfm-helm-public/heim/benchmark_output/runs/v1.1.0/{args.dataset}:skill={category},model={model},max_eval_instances=100/per_instance_stats.json"
                elif args.dataset == 'mscoco':
                    prompt_url = f"https://storage.googleapis.com/crfm-helm-public/heim/benchmark_output/runs/v1.1.0/{args.dataset}:model={model},max_eval_instances=100,groups={category}/instances.json"
                    score_url = f"https://storage.googleapis.com/crfm-helm-public/heim/benchmark_output/runs/v1.1.0/{args.dataset}:model={model},max_eval_instances=100,groups={category}/per_instance_stats.json"
                else:
                    prompt_url = f"https://storage.googleapis.com/crfm-helm-public/heim/benchmark_output/runs/v1.1.0/{args.dataset}:category={category},model={model},max_eval_instances=100/instances.json"
                    score_url = f"https://storage.googleapis.com/crfm-helm-public/heim/benchmark_output/runs/v1.1.0/{args.dataset}:category={category},model={model},max_eval_instances=100/per_instance_stats.json"

            # Send a GET request to the URL to download the JSON file
            response_prompt = requests.get(prompt_url)
            response_prompt.raise_for_status()  # Raise an exception for invalid HTTP responses

            # Send a GET request to the URL to download the JSON file
            response_score = requests.get(score_url)
            response_score.raise_for_status()  # Raise an exception for invalid HTTP responses


            with open(prompt_file, "wb") as f:
                f.write(response_prompt.content)

            with open(score_file, "wb") as f:
                f.write(response_score.content)

if __name__ == '__main__':
    main()
