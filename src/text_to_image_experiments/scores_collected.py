import os
import json
import argparse
import pickle

from utils import *

def winoground_mscoco_collect(args):
    results_exp_aesthetics = {}
    results_exp_clip = {}
    results_exp_lpips = {}
    results_exp_ssim = {}
    results_exp_psnr = {}
    results_exp_uiqi = {}
    results_max_aesthetics = {}
    results_max_clip = {}
    results_human_align = {}
    results_human_aesthetics = {}

    files = [f for f in sorted(os.listdir(os.path.join(args.save_dir, args.dataset)))]
    files.remove('.DS_Store') if '.DS_Store' in files else None
    count = 0
    for model_file in files:
        model = model_file[:-14]
        results_exp_aesthetics[model] = {}
        results_exp_clip[model] = {}
        results_exp_lpips[model] = {}
        results_exp_ssim[model] = {}
        results_exp_psnr[model] = {}
        results_exp_uiqi[model] = {}
        results_max_aesthetics[model] = {}
        results_max_clip[model] = {}
        results_human_align[model] = {}
        results_human_aesthetics[model] = {}

        rea, rec, rel, res, rep, reu, rma, rmc, rha, rhae = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        with open(os.path.join(args.save_dir, args.dataset, model_file), 'rb') as f:
            data = pickle.load(f)
        for item in data:
            rea[item], rec[item], rel[item], res[item], rep[item], reu[item], rma[item], rmc[item], rha[item], rhae[item] = \
                [], [], [], [], [], [], [], [], [], []
            for score in data[item][0]:
                rea[item].append(score[0])
                rma[item].append(score[1])
                rec[item].append(score[2])
                rmc[item].append(score[3])
                rha[item].append(score[4]) if score[4] != 'skip' else None
                rhae[item].append(score[5]) if score[5] != 'skip' else None
                rel[item].append(score[6])
                res[item].append(score[7])
                rep[item].append(score[8])
                reu[item].append(score[9]) if score[9] != 'skip' else None

        avg_el = sum([val for sublist in rel.values() for val in sublist]) / sum(
            len(sublist) for sublist in rel.values())
        avg_es = sum([val for sublist in res.values() for val in sublist]) / sum(
            len(sublist) for sublist in res.values())
        avg_ep = sum([val for sublist in rep.values() for val in sublist]) / sum(
            len(sublist) for sublist in rep.values())
        avg_eu = sum([val for sublist in reu.values() for val in sublist]) / sum(
            len(sublist) for sublist in reu.values())
        avg_ea = sum([val for sublist in rea.values() for val in sublist]) / sum(
            len(sublist) for sublist in rea.values())
        avg_ec = sum([val for sublist in rec.values() for val in sublist]) / sum(
            len(sublist) for sublist in rec.values())
        avg_ma = sum([val for sublist in rma.values() for val in sublist]) / sum(
            len(sublist) for sublist in rma.values())
        avg_mc = sum([val for sublist in rmc.values() for val in sublist]) / sum(
            len(sublist) for sublist in rmc.values())
        avg_ha = sum([val for sublist in rha.values() for val in sublist]) / sum(
            len(sublist) for sublist in rha.values())
        # avg_hae = sum([val for sublist in rhae.values() for val in sublist]) / sum(len(sublist) for sublist in rhae.values())
        try:
            avg_hae = sum([val for sublist in rhae.values() for val in sublist]) / sum(
                len(sublist) for sublist in rhae.values())
        except ZeroDivisionError:
            avg_hae = 0
            count = count + 1

        results_exp_lpips[model]["full"] = avg_el
        results_exp_ssim[model]["full"] = avg_es
        results_exp_psnr[model]["full"] = avg_ep
        results_exp_uiqi[model]["full"] = avg_eu
        results_exp_aesthetics[model]["full"] = avg_ea
        results_exp_clip[model]["full"] = avg_ec
        results_max_aesthetics[model]["full"] = avg_ma
        results_max_clip[model]["full"] = avg_mc
        results_human_align[model]["full"] = avg_ha
        results_human_aesthetics[model]["full"] = avg_hae

        results_exp_lpips[model]["classwise"] = rel
        results_exp_ssim[model]["classwise"] = res
        results_exp_psnr[model]["classwise"] = rep
        results_exp_uiqi[model]["classwise"] = reu
        results_exp_aesthetics[model]["classwise"] = rea
        results_exp_clip[model]["classwise"] = rec
        results_max_aesthetics[model]["classwise"] = rma
        results_max_clip[model]["classwise"] = rmc
        results_human_align[model]["classwise"] = rha
        results_human_aesthetics[model]["classwise"] = rhae

        results_human_align[model]["classwise"] = {key: value for key, value in
                                                   results_human_align[model]["classwise"].items() if value}
        results_human_aesthetics[model]["classwise"] = {key: value for key, value in
                                                        results_human_aesthetics[model]["classwise"].items() if value}
        results_exp_uiqi[model]["classwise"] = {key: value for key, value in
                                                results_exp_uiqi[model]["classwise"].items() if value}

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_aesthetics.json'), 'w') as f:
        json.dump(results_exp_aesthetics, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_max_aesthetics.json'), 'w') as f:
        json.dump(results_max_aesthetics, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_lpips.json'), 'w') as f:
        json.dump(results_exp_lpips, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_ssim.json'), 'w') as f:
        json.dump(results_exp_ssim, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_psnr.json'), 'w') as f:
        json.dump(results_exp_psnr, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_uiqi.json'), 'w') as f:
        json.dump(results_exp_uiqi, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_clip.json'), 'w') as f:
        json.dump(results_exp_clip, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_max_clip.json'), 'w') as f:
        json.dump(results_max_clip, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_human_align.json'), 'w') as f:
        json.dump(results_human_align, f, indent=4)

    if count != len(files):
        with open(os.path.join(args.save_dir, f'{args.dataset}_human_aesthetics.json'), 'w') as f:
            json.dump(results_human_aesthetics, f, indent=4)

def cub200_collect(args):
    results_exp_aesthetics = {}
    results_max_aesthetics = {}
    results_exp_clip = {}
    results_max_clip = {}
    results_exp_lpips = {}
    results_exp_ssim = {}
    results_exp_psnr = {}
    results_exp_uiqi = {}

    models = [f for f in sorted(os.listdir(os.path.join(args.save_dir, args.dataset)))]
    models.remove('.DS_Store') if '.DS_Store' in models else None

    for model_file in models:
        model = model_file[:-14]
        results_exp_aesthetics[model] = {}
        results_max_aesthetics[model] = {}
        results_exp_clip[model] = {}
        results_max_clip[model] = {}
        results_exp_lpips[model] = {}
        results_exp_ssim[model] = {}
        results_exp_psnr[model] = {}
        results_exp_uiqi[model] = {}

        with open(os.path.join(args.save_dir, args.dataset, model_file), 'rb') as f:
            data = pickle.load(f)

        rea, rma, rec, rmc, rel, res, rep, reu = {}, {}, {}, {}, {}, {}, {}, {}
        for item in data:
            rea[item], rma[item], rec[item], rmc[item], rel[item], res[item], rep[item], reu[
                item] = [], [], [], [], [], [], [], []
            for score in data[item][0]:
                rea[item].append(score[0])
                rma[item].append(score[1])
                rec[item].append(score[2])
                rmc[item].append(score[3])
                rel[item].append(score[4])
                res[item].append(score[5])
                rep[item].append(score[6])
                if score[7] == 'skip':
                    None
                else:
                    reu[item].append(score[7])

        avg_ea = sum([val for sublist in rea.values() for val in sublist]) / sum(
            len(sublist) for sublist in rea.values())
        avg_ma = sum([val for sublist in rma.values() for val in sublist]) / sum(
            len(sublist) for sublist in rma.values())
        avg_ec = sum([val for sublist in rec.values() for val in sublist]) / sum(
            len(sublist) for sublist in rec.values())
        avg_mc = sum([val for sublist in rmc.values() for val in sublist]) / sum(
            len(sublist) for sublist in rmc.values())
        avg_el = sum([val for sublist in rel.values() for val in sublist]) / sum(
            len(sublist) for sublist in rel.values())
        avg_es = sum([val for sublist in res.values() for val in sublist]) / sum(
            len(sublist) for sublist in res.values())
        avg_ep = sum([val for sublist in rep.values() for val in sublist]) / sum(
            len(sublist) for sublist in rep.values())
        avg_eu = sum([val for sublist in reu.values() for val in sublist]) / sum(
            len(sublist) for sublist in reu.values())

        results_exp_aesthetics[model]["full"] = avg_ea
        results_max_aesthetics[model]["full"] = avg_ma
        results_exp_clip[model]["full"] = avg_ec
        results_max_clip[model]["full"] = avg_mc
        results_exp_lpips[model]["full"] = avg_el
        results_exp_ssim[model]["full"] = avg_es
        results_exp_psnr[model]["full"] = avg_ep
        results_exp_uiqi[model]["full"] = avg_eu

        results_exp_aesthetics[model]["classwise"] = rea
        results_max_aesthetics[model]["classwise"] = rma
        results_exp_clip[model]["classwise"] = rec
        results_max_clip[model]["classwise"] = rmc
        results_exp_lpips[model]["classwise"] = rel
        results_exp_ssim[model]["classwise"] = res
        results_exp_psnr[model]["classwise"] = rep
        results_exp_uiqi[model]["classwise"] = reu

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_aesthetics.json'), 'w') as f:
        json.dump(results_exp_aesthetics, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_max_aesthetics.json'), 'w') as f:
        json.dump(results_max_aesthetics, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_clip.json'), 'w') as f:
        json.dump(results_exp_clip, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_max_clip.json'), 'w') as f:
        json.dump(results_exp_clip, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_lpips.json'), 'w') as f:
        json.dump(results_exp_lpips, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_ssim.json'), 'w') as f:
        json.dump(results_exp_ssim, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_psnr.json'), 'w') as f:
        json.dump(results_exp_psnr, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_uiqi.json'), 'w') as f:
        json.dump(results_exp_uiqi, f, indent=4)

def other_models_collect(args):
    results_exp_aesthetics = {}
    results_max_aesthetics = {}
    results_exp_clip = {}
    results_max_clip = {}
    results_human_align = {}
    results_human_aesthetics = {}

    models = [f for f in sorted(os.listdir(os.path.join(args.save_dir, args.dataset)))]
    models.remove('.DS_Store') if '.DS_Store' in models else None
    count = 0

    for model_file in models:
        model = model_file[:-14]
        results_exp_aesthetics[model] = {}
        results_max_aesthetics[model] = {}
        results_exp_clip[model] = {}
        results_max_clip[model] = {}
        results_human_align[model] = {}
        results_human_aesthetics[model] = {}

        with open(os.path.join(args.save_dir, args.dataset, model_file), 'rb') as f:
            data = pickle.load(f)

        rea, rma, rec, rmc, rha, rhae = {}, {}, {}, {}, {}, {}
        for item in data:
            rea[item], rma[item], rec[item], rmc[item], rha[item], rhae[item] = [], [], [], [], [], []
            for score in data[item][0]:
                rea[item].append(score[0])
                rma[item].append(score[1])
                rec[item].append(score[2])
                rmc[item].append(score[3])
                rha[item].append(score[4]) if score[4] != 'skip' else None
                rhae[item].append(score[5]) if score[5] != 'skip' else None

        avg_ea = sum([val for sublist in rea.values() for val in sublist]) / sum(
            len(sublist) for sublist in rea.values())
        avg_ma = sum([val for sublist in rma.values() for val in sublist]) / sum(
            len(sublist) for sublist in rma.values())
        avg_ec = sum([val for sublist in rec.values() for val in sublist]) / sum(
            len(sublist) for sublist in rec.values())
        avg_mc = sum([val for sublist in rmc.values() for val in sublist]) / sum(
            len(sublist) for sublist in rmc.values())
        avg_ha = sum([val for sublist in rha.values() for val in sublist]) / sum(
            len(sublist) for sublist in rha.values())
        # avg_hae = sum([val for sublist in rhae.values() for val in sublist]) / sum(len(sublist) for sublist in rhae.values())

        try:
            avg_hae = sum([val for sublist in rhae.values() for val in sublist]) / sum(
                len(sublist) for sublist in rhae.values())

        except ZeroDivisionError:
            avg_hae = 0
            count = count + 1

        results_exp_aesthetics[model]["full"] = avg_ea
        results_max_aesthetics[model]["full"] = avg_ma
        results_exp_clip[model]["full"] = avg_ec
        results_max_clip[model]["full"] = avg_mc
        results_human_align[model]["full"] = avg_ha
        results_human_aesthetics[model]["full"] = avg_hae

        # results_exp_clip[model]["full"] = avg_ec

        results_exp_aesthetics[model]["classwise"] = rea
        results_max_aesthetics[model]["classwise"] = rma
        results_exp_clip[model]["classwise"] = rec
        results_max_clip[model]["classwise"] = rmc
        results_human_align[model]["classwise"] = rha
        results_human_aesthetics[model]["classwise"] = rhae

        # results_exp_clip[model]["classwise"] = rec
        results_human_align[model]["classwise"] = {key: value for key, value in
                                                   results_human_align[model]["classwise"].items() if value}
        results_human_aesthetics[model]["classwise"] = {key: value for key, value in
                                                        results_human_aesthetics[model]["classwise"].items() if value}

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_aesthetics.json'), 'w') as f:
        json.dump(results_exp_aesthetics, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_max_aesthetics.json'), 'w') as f:
        json.dump(results_max_aesthetics, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_exp_clip.json'), 'w') as f:
        json.dump(results_exp_clip, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_max_clip.json'), 'w') as f:
        json.dump(results_max_clip, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_human_align.json'), 'w') as f:
        json.dump(results_human_align, f, indent=4)

    if count != len(models):
        with open(os.path.join(args.save_dir, f'{args.dataset}_human_aesthetics.json'), 'w') as f:
            json.dump(results_human_aesthetics, f, indent=4)
