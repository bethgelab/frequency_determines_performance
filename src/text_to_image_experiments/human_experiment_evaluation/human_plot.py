import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
import torch
import itertools
import pickle
import seaborn as sns
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["grid.linestyle"] = ':'
rcParams["xtick.direction"] = 'in'
rcParams["ytick.direction"] = 'in'
rcParams["legend.fontsize"] = 11
rcParams["axes.labelsize"] = 18
rcParams["axes.titlesize"] = 20
rcParams["xtick.labelsize"] = 15
rcParams["ytick.labelsize"] = 15

concept = 'entity'
entity_path = 'entities/P31_Q5_50000'
participants = ['Nicholas','Daphne','Matthew','Milad','Yash','Katherine']
version = 1
only_all = True
only_participants = False
only_participant = 'Katherine'
headshot_of = True
headshot = False
with open(os.path.join(f'{entity_path}/cnt_space.npy'), 'rb') as f:
    counts = np.load(f)
with open(os.path.join(f'{entity_path}.txt'), 'r') as f:
    concepts = np.array(f.read().split('\n'))
df_o = pd.DataFrame({concept: concepts, 'count': counts})
def get_buckets(df_o):
    if df_o.columns[0] == 'entity':
        df = df_o[df_o['entity'].str.contains(' ')] # only entities with spaces
    else:
        df = df_o
    df = df.sort_values("count", ascending=False)
    bucket_size = 1000
    bucket_min_count = 100
    buckets = []
    current_bucket = []
    current_bucket_count = 0
    current_bucket_min = float('inf')
    current_bucket_max = float('-inf')
    for i in range(len(df)):
        count = df.iloc[i]["count"]
        if count < current_bucket_min and current_bucket_count >= bucket_min_count:
            buckets.append((current_bucket, current_bucket_count, current_bucket_min, current_bucket_max))
            current_bucket = []
            current_bucket_count = 0
            current_bucket_min = count
            current_bucket_max = count
        current_bucket_max = max(current_bucket_max, count)
        if current_bucket_max > 0:
            current_bucket.append(i)
            current_bucket_count += 1
            current_bucket_min = min(current_bucket_min, count)
        
        if current_bucket_count == bucket_size:
            buckets.append((current_bucket, current_bucket_count, current_bucket_min, current_bucket_max))
            current_bucket = []
            current_bucket_count = 0
            current_bucket_min = float('inf')
            current_bucket_max = float('-inf')
    total_count = 0
    bucket_dfs = []
    for i, (bucket_indices, count, minimum, maximum) in enumerate(buckets):
        bucket_dfs.append(df.iloc[bucket_indices])
        print(f"Bucket {i}: {count} {df_o.columns[0]}, minimum count = {minimum}, maximum count = {maximum}")
        total_count += count
    print(f"Total count: {total_count}")
    return buckets, bucket_dfs, df
buckets, bucket_dfs, df = get_buckets(df_o)
"""
bucket_results = []
for i, (bucket_indices, count, minimum, maximum) in enumerate(buckets):
    samples = os.listdir(f"/home/yashjsharma/yash-internship/buckets{'' if concept == 'entity' else '_act'}/bucket_{i}{'_hsof' if headshot_of else '_hs' if headshot else ''}/samples")
    text_probs = torch.load(f"/home/yashjsharma/yash-internship/buckets/bucket_{i}_hsof_clip.pt", 
                            map_location=torch.device('cpu'))[np.argsort(samples)]
    gt_idx = torch.tensor(df_o.index[df.index][bucket_indices].tolist()[:len(text_probs)]).to(text_probs.device)
    if len(text_probs) > len(gt_idx): #txt2image didn't error out on last batch
        try:
            assert len(text_probs) % len(gt_idx) == 0
        except AssertionError:
            print(i)
            continue
        multiple = len(text_probs) // len(gt_idx)
        gt_idx = gt_idx.repeat(multiple)
    bucket_results.append(torch.argmax(text_probs, dim=-1) == gt_idx).float()
print(bucket_results)
"""
#values_ = [bucket_accuracy]
bucket_accuracy_mturks = {}
if not only_participants:
    bucket_accuracy_mturks['all'] = defaultdict(list)
vers = f'_v{version}' if version else ''
for participant in participants:
    with open(f'{participant}{vers}_bucket_acc.pkl', 'rb') as f:
        bucket_accuracy_mturk = pickle.load(f)
        if bucket_accuracy_mturk:
            if not only_all:
                if not only_participant or participant == only_participant:
                    bucket_accuracy_mturks[participant] = bucket_accuracy_mturk
            if not only_participants:
                for key, value in bucket_accuracy_mturk.items():
                    bucket_accuracy_mturks['all'][key].extend(value)
ranges = [(x[2], x[3]) for i,x in enumerate(buckets) if i <= 45]
#assert len(ranges) == len(bucket_accuracy)
fig, ax = plt.subplots(figsize=(6.4, 4.8))

ax.set_xscale('log')
#labels = ['CLIP'] +  list(bucket_accuracy_mturks.keys())
labels = list(bucket_accuracy_mturks.keys())
#colors = ['black'] +  list(mcolors.TABLEAU_COLORS.values())
colors = list(mcolors.TABLEAU_COLORS.values())
patches = []

x_vals = []
y_vals = []

for j in range(len(labels)):
    for i, (xmin, xmax) in enumerate(ranges):
        center = (xmin + xmax) / 2
        if False:
        #if labels[j] == 'CLIP':
            val = bucket_accuracy[i]
        else:
            if bucket_accuracy_mturks[labels[j]][i]:
                val = np.array(bucket_accuracy_mturks[labels[j]][i], dtype=float).mean()
            else:
                val = None
        if val is not None:
            x_vals.append(center)
            y_vals.append(val)
            print(center, val)
            # ax.plot(center, val, 'o', color='indianred')
            # ax.plot([xmin, xmax], [val, val], '-', color='gray')
    patches.append(plt.Line2D([], [], color=colors[j], marker='o', label=labels[j])) 

A_1 = np.vstack([x_vals, np.ones(len(x_vals))]).T
m_1, c_1 = np.linalg.lstsq(A_1, y_vals, rcond=None)[0]
print(m_1)

x_line_1 = np.array([min(x_vals), max(x_vals)])
y_line_1 = m_1 * x_line_1 + c_1
# plt.plot(x_line_1, y_line_1, c='black', label='Fitted line 1', alpha=0.4, linewidth=2)


pairs = zip(x_vals, y_vals)

# Sort the pairs based on the elements from x_vals
sorted_pairs = sorted(pairs)

# Unzip the pairs back into two lists
sorted_x_vals, sorted_y_vals = zip(*sorted_pairs)

x_vals = sorted_x_vals[:-1]
y_vals = sorted_y_vals[:-1]

print(x_vals, 'xvals')
print(y_vals)

# plt.scatter(x_vals, y_vals)

nb = 5
    
bins = np.linspace(min(np.log(x_vals)), max(np.log(x_vals)), num=nb)
assigned_bins = np.digitize(np.log(x_vals), bins, right=True)

cumsums = [0]*len(bins)
cumcounts = [0]*len(bins)
cumarrs = {ab:[] for ab in assigned_bins}
for acc, xv, ab in zip(y_vals, x_vals, assigned_bins):
    cumsums[ab] += acc
    cumcounts[ab] += 1
    cumarrs[ab].append(acc)
cumaccs = [s/c if c > 0 else 0 for s, c in zip(cumsums, cumcounts)]
print(cumarrs.keys())
# cummeans = np.zeros(max(list(cumarrs.keys()))+1)
# cumstds = np.zeros(max(list(cumarrs.keys()))+1)

cummeans = np.zeros(nb)
cumstds = np.zeros(nb)

for key in cumarrs:
    try:
        cummeans[key] = np.mean(cumarrs[key])
    except IndexError as e:
        print("ERROR IN : {} {} {}".format(curr_model, pretrained_dataset, prompt_type))
        # return
    cumstds[key] = np.std(cumarrs[key])
print(cumsums)
print(cumcounts)
print(cumaccs)
print(cummeans)
print(cumstds)

import scipy
xxx, sig1 = scipy.stats.pearsonr(bins, cummeans)

print(xxx, 'pearson')

add1 = '**' if sig1<0.05 else ''

print(bins, 'bins')
plt.plot(np.exp(bins), cummeans, marker='o', linestyle='solid', alpha=0.6, markersize=8, linewidth=2, color='indianred', label=f'$\\rho$={xxx:.2f}'+add1)


ax.set_xlabel('Pretraining Concept Frequency')
ax.set_ylabel('Human Accuracy')
#ax.set_ylim([0, 1])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
#blue_patch = plt.Line2D([], [], color='blue', marker='o', label='human (unambig)')
# ax.legend(handles=patches)
plt.legend(loc='best', fontsize=16)
plt.tight_layout()
plt.grid()
plt.show()
plot_name = f"clip_count{vers}{''.join(list(bucket_accuracy_mturks.keys()))}{'_hsof' if headshot_of else '_hs' if headshot else ''}"
print(plot_name)
fig.savefig(plot_name, dpi=900)
for key in list(bucket_accuracy_mturks.keys()):
    num_elem_per_bucket = np.array([len(bucket_accuracy_mturks[key][i]) for i in range(len(bucket_accuracy_mturks[key]))])
    print(num_elem_per_bucket.min(), num_elem_per_bucket.mean(), num_elem_per_bucket.max())
