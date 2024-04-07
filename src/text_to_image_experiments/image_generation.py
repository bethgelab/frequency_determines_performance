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

    generator = torch.Generator(args.gpu).manual_seed(args.seed)
    batch = []
    concepts = []
    ids = []

    for num, (concept, prompt, count) in enumerate(prompt_list):
        if not os.path.isfile("{}/{}_{}.png".format(args.save_dir, concept, str(count))):
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
            elif args.pipe == 'dreamlike' or args.pipe == 'sd':
                images = pipe(batch, generator=generator, height=args.image_size, width=args.image_size,).images
                flush()

            for num, (count, concept, prompt, img) in enumerate(zip(ids, concepts, batch, images)):
                save_file = "{}/{}_{}.png".format(args.save_dir, concept, str(count))
                img.resize((args.image_size, args.image_size))
                img.save(save_file)
            batch = []
            ids = []
            concepts = []
            flush()

def main():
    parser = argparse.ArgumentParser(description="Image generation script for wag concepts")
    parser.add_argument(
        "--load_prompts", default="let_it_wag_datasets/let_it_wag_common_prompts_for_image_gen.pkl",
        help="path to prompts", type=str)

    parser.add_argument("--save_dir", default='let_it_wag_concepts/images/dreamlike/common/',
                        help="Path to save images", type=str)

    parser.add_argument("--cache_dir",
                        default='.cache/huggingface',
                        help="Cache directory for huggingface. Default is /home/.cache/huggingface/", type=str)

    parser.add_argument("--pipe",
                        default='dreamlike',
                        help="Which text-to-image model to use", type=str)

    parser.add_argument("--model_id",
                        default='dreamlike-art/dreamlike-photoreal-2.0',
                        help="Hugging face model id to use", type=str)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=1023, type=int)

    args = parser.parse_args()
    set_seed(args.seed)
    #check if folder exists, if not create folder
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

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

    elif args.pipe == 'dreamlike' or args.pipe == 'sd':
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