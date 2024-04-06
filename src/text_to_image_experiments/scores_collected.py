import json
import os
import pickle


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


root_dir = 'results'
dataset = 'draw_bench'

results_exp_aesthetics = {}
results_max_aesthetics = {}
results_exp_clip = {}
results_max_clip = {}
results_human_align = {}
results_human_aesthetics = {}

models = [f for f in sorted(os.listdir(os.path.join(root_dir, dataset)))]
models.remove('.DS_Store') if '.DS_Store' in models else None

for model_file in models:
    model = model_file[:-14]
    results_exp_aesthetics[model] = {}
    results_max_aesthetics[model] = {}
    results_exp_clip[model] = {}
    results_max_clip[model] = {}
    results_human_align[model] = {}
    results_human_aesthetics[model] = {}

    with open(os.path.join(root_dir, dataset, model_file), 'rb') as f:
        data = pickle.load(f)

    rea, rma, rec, rmc, rha, rhae = {}, {}, {}, {}, {}, {}
    for item in data:
        rea[item] = data[item][0]
        rma[item] = data[item][1]
        rec[item] = data[item][2]
        rmc[item] = data[item][3]
        if dataset == 'coco':
            rha[item] = data[item][4]
            rhae[item] = data[item][5]
    avg_ea = sum(rea.values()) / len(rea)
    avg_ma = sum(rma.values()) / len(rma)
    avg_ec = sum(rec.values()) / len(rec)
    avg_mc = sum(rmc.values()) / len(rmc)

    results_exp_aesthetics[model]["full"] = avg_ea
    results_max_aesthetics[model]["full"] = avg_ma
    results_exp_clip[model]["full"] = avg_ec
    results_max_clip[model]["full"] = avg_mc

    results_exp_aesthetics[model]["classwise"] = rea
    results_max_aesthetics[model]["classwise"] = rma
    results_exp_clip[model]["classwise"] = rec
    results_max_clip[model]["classwise"] = rmc

    if dataset == 'coco':
        avg_ha = sum(rha.values()) / len(rha)
        avg_hae = sum(rhae.values()) / len(rhae)
        results_human_align[model]["full"] = avg_ha
        results_human_aesthetics[model]["full"] = avg_hae
        results_human_align[model]["classwise"] = rha
        results_human_aesthetics[model]["classwise"] = rhae

with open(os.path.join(root_dir, f'{dataset}_exp_aesthetics.json'), 'w') as f:
    json.dump(results_exp_aesthetics, f, indent=4)

with open(os.path.join(root_dir, f'{dataset}_max_aesthetics.json'), 'w') as f:
    json.dump(results_max_aesthetics, f, indent=4)

with open(os.path.join(root_dir, f'{dataset}_exp_clip.json'), 'w') as f:
    json.dump(results_exp_clip, f, indent=4)

with open(os.path.join(root_dir, f'{dataset}_max_clip.json'), 'w') as f:
    json.dump(results_max_clip, f, indent=4)

if dataset == 'coco':
    with open(os.path.join(root_dir, f'{dataset}_human_align.json'), 'w') as f:
        json.dump(results_human_align, f, indent=4)

    with open(os.path.join(root_dir, f'{dataset}_human_aesthetics.json'), 'w') as f:
        json.dump(results_human_aesthetics, f, indent=4)