# This is matched to the metadata csv from https://drive.google.com/file/d/1bakfXSS5Dcbf6JtQ-sNEH-ghqYlNZxmd/view?usp=sharing
# The above metadata is from https://github.com/mlfoundations/clip_quality_not_quantity/tree/main
wget https://huggingface.co/datasets/vishaal27/YFCC15M_page_and_download_urls/resolve/main/yfcc15m_final_split_pageandimageurls.csv
img2dataset --url_list yfcc15m_final_split_pageandimageurls.csv --input_format "csv" --output_format webdataset --output_folder images --processes_count 2 --thread_count 8 --resize_mode no --enable_wandb True
