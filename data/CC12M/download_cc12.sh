# Refer to the link for more details-- https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md
wget -c https://storage.googleapis.com/conceptual_12m/cc12m.tsv
for i in {00000..01099}; do wget -c https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/$i.tar; done
