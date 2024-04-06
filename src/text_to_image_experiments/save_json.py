import requests

categories = ["Animals","Artifacts","Arts","Food","Illustrations","Indoor","Outdoor","People","Produce",
              "Vehicles","World"]
models = ["DeepFloyd_IF-I-L-v1.0","DeepFloyd_IF-I-M-v1.0","DeepFloyd_IF-I-XL-v1.0","adobe_giga_gan","craiyon_dalle-mega",
          "craiyon_dalle-mini","huggingface_dreamlike-diffusion-v1-0","huggingface_dreamlike-photoreal-v2-0",
          "huggingface_openjourney-v1-0", "huggingface_openjourney-v2-0", "huggingface_promptist-stable-diffusion-v1-4",
          "huggingface_redshift-diffusion", "huggingface_stable-diffusion-safe-max", "huggingface_stable-diffusion-safe-medium",
          "huggingface_stable-diffusion-safe-strong", "huggingface_stable-diffusion-safe-weak", "huggingface_stable-diffusion-v1-4",
          "huggingface_stable-diffusion-v1-5", "huggingface_stable-diffusion-v2-1-base", "huggingface_stable-diffusion-v2-base",
          "huggingface_vintedois-diffusion-v0-1", "kakaobrain_mindall-e", "lexica_search-stable-diffusion-1.5"]
save_dir = "data/parti/"
for category in categories:
    for model in models:
        prompt_dir = os.path.join(save_dir,model,'prompt')
        score_dir = os.path.join(save_dir,model,'score')
        [os.makedirs(directory, exist_ok=True) for directory in [prompt_dir,score_dir]]

        prompt_file = os.path.join(prompt_dir,category.lower()+'.json')
        score_file = os.path.join(score_dir,category.lower()+'.json')

        prompt_url = f'https://nlp.stanford.edu/helm/v1.1.0/benchmark_output/runs/latest/parti_prompts:category={category},model={model},max_eval_instances=100/instances.json'
        score_url = f'https://nlp.stanford.edu/helm/v1.1.0/benchmark_output/runs/latest/parti_prompts:category={category},model={model},max_eval_instances=100/per_instance_stats.json'

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