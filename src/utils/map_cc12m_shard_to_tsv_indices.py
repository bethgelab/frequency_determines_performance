# The text search results are computed using the indices from the cc12m.tsv file.
# Therefore, all the text search results contain as the outputs indices (row-numbers) from the cc12m.tsv file
# 
# However, the image search results are computed using the indices from the cc12m shards (1100 shards).
# We therefore have to map the image indices (from the shards) back to the text indices (from the cc12m.tsv)
# for taking set intersection between the image and text indices.
# For this, we can take the corresponding caption for each image index that we hit, find that caption in the cc12m.tsv file
# and then map the index of this hit as the row-number of that caption in the cc12m.tsv files
import os
import pickle
import pandas as pd
from tqdm import tqdm
import tarfile
import re
import argparse

rampp_pkl_files_path = 'rampp_outputs/cc12m'
csv_file_path = '../../data/cc12m/cc12m.tsv'
shards_path = '../../data/cc12m'

def create_index_to_text_dict(tar_path):
    index_to_text = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.txt'):
                index = int(re.match(r"\d+", member.name).group())
                file = tar.extractfile(member)
                if file:
                    text_content = file.read().decode('utf-8')
                    index_to_text[index] = text_content
    return index_to_text

def rename_pkl_shards():
    pkl_files = sorted(os.listdir(rampp_pkl_files_path))
    for f in tqdm(pkl_files, ascii=True, total=len(pkl_files)):
        with open(os.path.join(rampp_pkl_files_path, f), 'rb') as f_:
            pkl_index = int(f.split('.')[0].split('_')[-1])
            offset = pkl_index * 10000
            x = pickle.load(f_)
            fnames = x['filenames']
            outnames = []
            for old_name in sorted(fnames):
                old_int = int(old_name.split('.')[0])
                out_int = offset + old_int
                out_name_ = str(out_int).zfill(8)
                outnames.append('{}.jpg'.format(out_name_))
        with open(os.path.join(rampp_pkl_files_path, f.replace('cc12m', 'cc12m_new')), 'wb') as xxx:
            pickle.dump({'confidence_threshold': x['confidence_threshold'], 'filenames': outnames, 'tags': x['tags'], 'probs': x['probs']}, xxx)

def get_shard_id_to_csv_mapping(chunk_idx):
    df = pd.read_csv(csv_file_path, sep='\t', header=None)
    csv_caption_to_index = {caption: index for index, caption in enumerate(df[1])}
    del df

    mapping_file = os.path.join('../../data/cc12m/mappings_from_shard_ids_to_csv_ids/cc12m_shard_ids_to_csv_id_mapping_{}.pkl'.format(chunk_idx))
    mapping_shard_id_to_csv = {}

    pkl_files = sorted(os.listdir(rampp_pkl_files_path))
    for f in tqdm(pkl_files, ascii=True, total=len(pkl_files)):
        pkl_index = int(f.split('.')[0].split('_')[-1])
        if pkl_index != chunk_idx:
            continue
        offset = pkl_index * 10000
        with open(os.path.join(rampp_pkl_files_path, f), 'rb') as f_:
            x = pickle.load(f_)
        fnames = x['filenames']
        shard_id_to_caption = create_index_to_text_dict(os.path.join(shards_path, 'shards', '{}.tar'.format(str(pkl_index).zfill(5))))
        for name_index, name in enumerate(sorted(fnames)):
            old_int = int(name.split('.')[0])
            offsetted_index = old_int - offset
            mapping_shard_id_to_csv[name] = csv_caption_to_index[shard_id_to_caption[offsetted_index]]
        with open(mapping_file, 'wb') as out_:
            pickle.dump(mapping_shard_id_to_csv, open(mapping_file, 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()

    # this function is used for renaming the saved filenames in the rampp output pkl files to follow the offset + index format to retrieve captions and tags
    rename_pkl_shards()
    # get_shard_id_to_csv_mapping(args.chunk_idx)