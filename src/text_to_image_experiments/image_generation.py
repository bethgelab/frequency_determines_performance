import os
import argparse
import torch
import random
import numpy as np
import pickle

from diffusers import StableDiffusionXLPipeline,DiffusionPipeline,StableDiffusionPipeline
# from huggingface_hub import login

from utils import *


def generate_images(args, pipe, refiner, prompt_list):
    added_prompt = "best quality, HD, 32K, high focus, dramatic lighting, ultra-realistic, high detailed photography"
    negative_prompt = 'nude, naked, text, digits, worst quality, blurry, morbid, poorly drawn face, bad anatomy,distorted face, disfiguired, bad hands, missing fingers,cropped, deformed body, bloated, ugly, unrealistic'

    generator = torch.Generator(args.gpu).manual_seed(args.seed)

    batch = []
    concepts = []
    ids = []
    prompt_list = [['ray', "A majestic manta ray glides gracefully through a clear blue ocean, sunlight filters down through the water, illuminating the plankton swirling around it.",1],
                   ['ray', "A dark stingray glides silently through a sunlit coral reef, casting a fleeting shadow as it searches for food among the colorful fish.",2],
                   ['ray', "A majestic ray gliding through vibrant coral reefs, casting shadows on the ocean floor.",3],
                   ['ray', "A solitary ray elegantly dancing amidst swaying sea plants, its sleek silhouette contrasting against the azure backdrop.", 4]]

    for num, (concept, prompt, count) in enumerate(prompt_list):
        if not os.path.isfile("{}/{}_{}.png".format(args.save_path, concept, str(count))):
            batch.append(prompt)
            concepts.append(concept)
            ids.append(count)
            if len(batch) < args.batch_size and (num + 1) < len(prompt_list):
                continue

            #the last batch might not have the same size as batch_size so we use len(batch) instead of batch_size
            if args.pipe == 'sdxl':
                images = pipe(batch, generator=generator, height = args.image_size, width = args.image_size,denoising_end=0.8,
                              output_type="latent",).images

                images = refiner(prompt=batch, denoising_start=0.8,image=images).images
            elif args.pipe == 'dreamlike' or args.pipe == 'openjourney' or args.pipe == 'sd':
                images = pipe(batch, generator=generator, height=args.image_size, width=args.image_size,).images
                flush()

            elif args.pipe == 'deep_floyd':
                # text embeds
                prompt_embeds, negative_embeds = pipe[0].encode_prompt(batch)

                # stage 1
                images = pipe[0](
                    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator,
                    output_type="pt").images
                # del pipe[0]
                flush()
                # stage 2
                images = pipe[1](
                    image=images, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type="pt").images
                # del pipe[1]
                flush()
                # stage 3
                images = pipe[2](prompt=batch, image=images, generator=generator).images
                # del pipe[2]
                flush()

            for num, (count, concept, prompt, img) in enumerate(zip(ids, concepts, batch, images)):
                save_file = "{}/{}_{}.png".format(args.save_path, concept, str(count))
                img.resize((args.image_size, args.image_size))
                img.save(save_file)
            batch = []
            ids = []
            concepts = []
            flush()

def main():
    parser = argparse.ArgumentParser(description="Image generation script for wag concepts")
    parser.add_argument(
        "--load_prompts", default="/mnt/qb/work/bethge/bkr536/eccv/let_it_wag_concepts/head_prompts.pkl",
        help="path to prompts", type=str)

    parser.add_argument("--save_path", default='/mnt/qb/work/bethge/bkr536/eccv/let_it_wag_concepts/images/dreamlike/head/',
                        help="Path to save images", type=str)

    parser.add_argument("--cache_dir",
                        default='/mnt/qb/work/bethge/bkr536/.cache/huggingface',
                        help="Cache directory for huggingface. Default is /home/.cache/huggingface/", type=str)

    parser.add_argument("--pipe",
                        default='dreamlike',
                        help="Which text-to-image model to use", type=str)

    parser.add_argument("--model_id",
                        default='dreamlike-art/dreamlike-photoreal-2.0',
                        help="Hugging face model id to use", type=str)
    # OR dreamlike-art/dreamlike-photoreal-2.0
    # stabilityai/stable-diffusion-2-1
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=1023, type=int)

    args = parser.parse_args()
    set_seed(args.seed)
    #check if folder exists, if not create folder
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(args.load_prompts, 'rb') as f:
        results = pickle.load(f)

    print("Loading pipeline")

    if args.pipe == 'sdxl':
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                         torch_dtype=torch.float16, use_safetensors=True,
                                                         cache_dir = args.cache_dir).to("cuda")
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2, vae=pipe.vae, torch_dtype=torch.float16, use_safetensors=True,
            variant="fp16",
            cache_dir=args.cache_dir).to("cuda")
        refiner.enable_vae_slicing()
        pipe.enable_vae_slicing()

    elif args.pipe == 'deep_floyd':
        # stage 1
        stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16,cache_dir=args.cache_dir).to("cuda")
        stage_1.enable_model_cpu_offload()

        # stage 2
        stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16,cache_dir=args.cache_dir).to("cuda")
        stage_2.enable_model_cpu_offload()

        # stage 3
        safety_modules = {
            "feature_extractor": stage_1.feature_extractor,
            "safety_checker": stage_1.safety_checker,
            "watermarker": stage_1.watermarker,
        }
        stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16,cache_dir=args.cache_dir).to("cuda")
        stage_3.enable_model_cpu_offload()
        stage_3.enable_attention_slicing()
        pipe = [stage_1, stage_2, stage_3]

    elif args.pipe == 'dreamlike' or args.pipe == 'openjourney' or args.pipe == 'sd':
        pipe = StableDiffusionPipeline.from_pretrained(args.model_id,torch_dtype=torch.float16, use_safetensors=True,
                                                         cache_dir = args.cache_dir).to("cuda")

    prompt_list = []
    for concept in results:
        for i, prompt in enumerate(results[concept]):
            prompt_list.append([concept.lower().replace(' ', '_'), prompt, i + 1])

    if args.pipe == 'sdxl':
        generate_images(args, pipe, refiner, prompt_list)
    else:
        generate_images(args, pipe, None, prompt_list)
    print("Done")

if __name__ == '__main__':
    main()
