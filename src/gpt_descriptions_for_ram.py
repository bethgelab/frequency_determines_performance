from openai import OpenAI
import json
from tqdm import tqdm
import argparse
from downstream_dataloader import DataLoader
import os
import numpy as np

# set openai key as env variable by:
# export OPENAI_API_KEY=sk-xxx
parser = argparse.ArgumentParser(
    description='Generate LLM tag descriptions for RAM++ open-set recognition')
parser.add_argument('--output_file_path',
                    help='save path of llm tag descriptions',
                    default='/home/vu214/blind_name_only_transfer/gpt_descriptions')
# for zero-shot classification datasets, pass in dataset name
# for retrieval tasks, pass in `retrieval`, it will read from `ram_model/ram/data/ram_tag_list_for_retrieval_tasks_non_overlapping_with_zs_concepts.txt`
# for t2i tasks, pass in `t2i`, it will read from `ram_model/ram/data/ram_tag_list_for_t2i_tasks_non_overlapping_with_zs_concepts.txt` or the updated `ram_model/ram/data/ram_tag_list_for_updated_t2i_tasks_non_overlapping_with_previous_concepts.txt`
parser.add_argument('--dataset', default='t2i')

def analyze_tags(tag, client):

    # Generate LLM tag descriptions
    llm_prompts = [ f"Describe concisely what a(n) {tag} looks like:", \
                    f"How can you identify a(n) {tag} concisely?", \
                    f"What does a(n) {tag} look like concisely?",\
                    f"What are the identifying characteristics of a(n) {tag}:", \
                    f"Please provide a concise description of the visual characteristics of {tag}:"]

    results = {}
    result_lines = []

    result_lines.append(f"a photo of a {tag}.")

    for llm_prompt in tqdm(llm_prompts, ascii=True, total=len(llm_prompts)):

        response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "assistant", "content": llm_prompt}],
        temperature=0.99,
        max_tokens=77,
        n=10,
        stop=None
        )

        for item in response.choices:
            result_lines.append(item.message.content.strip())
        results[tag] = result_lines
    return results

if __name__ == "__main__":

    args = parser.parse_args()

    # dummy parameters for dataloader
    args.val_batch_size = 64
    args.train_batch_size = 256
    args.k_shot = 1

    client = OpenAI()

    # get classnames
    if args.dataset == 'retrieval':
        with open('../ram_model/ram/data/ram_tag_list_for_retrieval_tasks_non_overlapping_with_zs_concepts.txt', 'r') as f:
            categories = [z.strip('\n') for z in f.readlines()]
    elif args.dataset == 't2i':
        with open('../ram_model/ram/data/ram_tag_list_for_updated_t2i_tasks_non_overlapping_with_previous_concepts.txt', 'r') as f:
            categories = [z.strip('\n') for z in f.readlines()]
    else:
        data_loader = DataLoader(args, None)
        _, _, _, _, num_classes, string_classnames = data_loader.load_dataset()    
        categories = string_classnames
        if args.dataset == 'cifar10':
            categories[0] = 'airplane'

    output_file_path = os.path.join(args.output_file_path, '{}_descriptions.json'.format(args.dataset))

    tag_descriptions = {}
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            tag_descriptions = json.load(f)

    for tag in categories:
        print('Descriptions for category: {}'.format(tag))
        if tag in tag_descriptions:
            print('Already retrieved descriptions for {}'.format(tag))
            continue
        result = analyze_tags(tag, client)
        tag_descriptions[tag] = result

        with open(output_file_path, 'w') as w:
            json.dump(tag_descriptions, w, ensure_ascii=False, indent=4)
