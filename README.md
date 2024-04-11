# Pretraining Concept Frequency determines Multimodal Model Performance [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![PyTorch](https://img.shields.io/badge/PyTorch-grey.svg?logo=PyTorch)](https://pytorch.org/blog/pytorch-1.9-released/) [![Let-It-Wag Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Let_It_Wag!-blue)](https://huggingface.co/datasets/bethgelab/Let-It-Wag) [![Data Artefacts](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data_Artefacts-red)](https://huggingface.co/datasets/bethgelab/frequency_determines_performance) [![Paper](http://img.shields.io/badge/paper-arxiv.2404.04125-B31B1B.svg)](https://arxiv.org/abs/2404.04125)

This is the official codebase for the paper, ["No Zero-Shot Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance"](https://arxiv.org/abs/2404.04125).
Authors: [Vishaal Udandarao*](http://vishaal27.github.io/), [Ameya Prabhu*](https://drimpossible.github.io/), [Adhiraj Ghosh](https://adhirajghosh.github.io/), [Yash Sharma](https://www.yash-sharma.com/), [Philip H.S. Torr](https://scholar.google.com/citations?user=kPxa2w0AAAAJ&hl=en), [Adel Bibi](https://www.adelbibi.com/), [Samuel Albanie](http://samuelalbanie.com/) and [Matthias Bethge](https://scholar.google.com/citations?user=0z0fNxUAAAAJ). 

## Getting started
All our code was tested on Python 3.8.13 with Pytorch 2.0.1+cu117. Ideally, most of our scripts require access to a single GPU (uses `.cuda()` for inference). Inference can also be done on CPUs with minimal changes to the scripts.

#### Setting up environments
We recommend setting up a python virtual environment and installing all the requirements. Please follow these steps to set up the project folder correctly:

```bash
git clone https://github.com/bethgelab/frequency_determines_performance.git
cd frequency_determines_performance

conda create --name env python=3.8 -y
conda activate env
pip install -r requirements.txt
```

#### Setting up datasets
We provide detailed instructions on how to set up both pretraining and downstream test datasets in [`data/README.md`](https://github.com/bethgelab/frequency_determines_performance/blob/main/data/README.md).

## Exploring and Running Concept Frequency Analyses
We now describe how to run the different analyses in the paper independently below.

#### Extracting concepts
For the zero-shot classification tasks, the concepts are simply the classnames. For the retrieval tasks, we present how to run the script for extracting the concepts here:
```bash
python src/downstream_retrieval_extract_concepts.py --dataset <coco/flickr> --batch_size <bs>
```
The `batch_size` parameter is used for processing the documents in the Spacy pipeline.
For the text-to-image generation tasks, we process the concepts similarly to the retrieval pipeline followed by manual curation.

#### Constructing text index
To construct the inverted index for all the text captions of a given pretraining dataset, run this script which will produce individual chunk-wise inverted indexes:
```bash
python src/text_search_inverted_index_get_word_dictionaries.py --dataset <CC3M/CC12M/...> --path <path_to_dataset> --save_path <path_to_save_index> --batch_size <bs> --chunk_idx <chunk_index> --num_chunks <num_chunks>
```
Again, `batch_size` determines the processing batch size used in the Spacy pipeline, `chunk_idx` and `num_chunks` determine how many captions to process parallely and batch together.
Once the individual indexes are created, run this to merge them:
```bash
python src/text_search_inverted_index_combine_dictionaries.py --dataset <CC3M/CC12M/...> --save_filepath <path_to_save_index> --total_chunks <num_chunks_in_total_to_merge>
```
This script takes all `total_chunks` number of chunked inverted indexes, and merges them into one large text inverted index.

#### Constructing image index
For constructing the image index, we utilise the [RAM++ model](https://github.com/xinyu1205/recognize-anything). To run effectively, the model takes as input a list of concepts that it has to tag each image with. Additionally, it also requires a list of GPT-generated descriptions for each concept. We provide a script to do this here:
```bash
python src/gpt_descriptions_for_ram.py --dataset <coco/flickr/t2i/birdsnap/...>
```
For convenience, we provide all our generated description lists in [`gpt_descriptions`](https://github.com/bethgelab/frequency_determines_performance/tree/main/gpt_descriptions). The combined json file with all 4,029 descriptions that we use for the RAM++ model inference is here: [`gpt_descriptions/rampp_overall.json`](https://github.com/bethgelab/frequency_determines_performance/blob/main/gpt_descriptions/rampp_overall.json).
Once we have this json file of descriptions, we can run inference with the RAM++ model on all the images of a pretraining dataset, tagging each pretraining image with concepts from the 4,029 list:
```bash
python ram_model/image_search_run_rampp.py --pt_dataset <cc3m/cc12m/laion400m/...> --load_path <path_to_dataset_tars> --chunk_idx <tar_number_to_process> --batch_size_rampp <bs> --confidence_threshold <confidence_threshold_for_concept_consideration> --pretrained <path_to_ram++_checkpoint> --cache_dir <path_to_model_cache> --features_dir <path_to_store_features> --results_dir <path_to_store_results>
```
The `confidence_threshold` parameter controls the threshold above which we consider a concept i.e., the RAM++ model when ran on a pretraining image produces 4,029 logits which can be converted to probability values. The `confidence_threshold` determines the threshold above which if a particular concept's probability is, we consider it to be a part of that pretraining image's tag set.

Once the inference script above is run, we can construct the full image index (dictionary of size 4,029) using:
```bash
python src/image_search_inverted_index_creation.py --pt_dataset <cc3m/cc12m/laion400m/...> --start_index_id <start_index_id> --end_index_id <end_index_id> --cache_dir <path_to_model_cache> --features_dir <path_to_store_features> --results_dir <path_to_store_results>
```

#### Concept frequency estimation
Having constructed both the image and text indexes for efficient frequency estimation, we can run the image-only, text-only and image-text combined searches directly using:
```bash
# for image search
python src/image_search_matches_inverted_index.py --pt_dataset <cc3m/cc12m/laion400m/...> --threshold <confidence_threshold_for_ram++> --downstream_dataset <coco/flickr/cifar10/t2i/...> --cache_dir <path_to_model_cache> --features_dir <path_to_store_features> --results_dir <path_to_store_results>

# for text search
python src/text_search_matches_inverted_index.py --pt_dataset <cc3m/cc12m/laion400m/...> --search_method lemmatized --downstream_dataset <coco/flickr/cifar10/t2i/...> --do_chunked_search True

# for image-text search
python src/integrated_search_matches_inverted_index.py --pt_dataset <cc3m/cc12m/laion400m/...> --downstream_dataset <coco/flickr/cifar10/t2i/...>
```
For more details on the parameters of each search script, please see the comments within each script directly. These scripts should write to the results folder directly. For convenience, we have included all our search results across all concepts, pretraining datasets and downstream datasets in this folder: [`search_counts`](https://github.com/bethgelab/frequency_determines_performance/tree/main/search_counts). These counts are directly used for the plots in our paper.

#### Running downstream evaluations
We provide scripts to evaluate all models on the downstream zero-shot classification and retrieval tasks:
```bash
# zero-shot classification
python src/zero_shot_eval.py --backbone <RN50/RN101/ViT-B-32/...> --pretraining <cc3m/cc12m/yfcc15m/...> --text_prompts <simple/class/ensemble> --dataset <cifar10/imagenet/...> --cache_dir <path_to_model_cache> --features_dir <path_to_store_features> --results_dir <path_to_store_results>

# retrieval
python src/retrieval_eval.py --dataset <coco/flickr> --backbone <RN50/RN101/ViT-B-32/...> --pretraining <cc3m/cc12m/yfcc15m/...> --cache_dir <path_to_model_cache> --features_dir <path_to_store_features> --results_dir <path_to_store_results>
```
The results will be saved directly as json files. For convenience, we have also included all our result jsons here: [[zero-shot classification](https://github.com/bethgelab/frequency_determines_performance/tree/main/zero_shot_evaluations)] / [[retrieval](https://github.com/bethgelab/frequency_determines_performance/tree/main/retrieval_evaluations)] / [[text-to-image generation](https://github.com/bethgelab/frequency_determines_performance/tree/main/t2i_evaluations)].

#### Plotting
All the plots in our paper can be reproduced using the notebooks provided in the [`notebooks`](https://github.com/bethgelab/frequency_determines_performance/tree/main/notebooks) folder. The notebooks should be self-explanatory, please open an issue if something is unclear.

#### Text-to-image generation experiments
We provide all details on data collection, processing and running experiments for our text-to-image generation experiments here: [https://github.com/bethgelab/frequency_determines_performance/blob/main/src/text_to_image_experiments/README.md](https://github.com/bethgelab/frequency_determines_performance/blob/main/src/text_to_image_experiments/README.md). Please follow the instructions in that folder to reproduce all the text to image generation data collection steps.

#### Additional insights
You can reproduce our long-tailed nature plot from the [`notebooks/long-tailed-nature.ipynb`](https://github.com/bethgelab/frequency_determines_performance/blob/main/notebooks/long-tailed-nature.ipynb) notebook. The correlations between different pretraining dataset concept distributions numbers can be reproduced here: [`notebooks/correlations_between_pretraining_frequencies.ipynb`](https://github.com/bethgelab/frequency_determines_performance/blob/main/notebooks/correlations_between_pretraining_frequencies.ipynb). Finally, for quantifying the misalignment ratio, this is the script we use:
```bash
python src/misalignment_degree_quantification.py --pt_dataset <cc3m/cc12m/laion400m/...> --cache_dir <path_to_model_cache> --features_dir <path_to_store_features> --results_dir <path_to_store_results>
```
We also provide the logs of the misalignment ratios we computed for all datasets here: [`misalignment_metrics`](https://github.com/bethgelab/frequency_determines_performance/tree/main/misalignment_metrics).

## Releasing data research artefacts
We release all our processed pretraining and downstream data files (features, indexes etc) (about 300GB) on Huggingface here: [https://huggingface.co/datasets/bethgelab/frequency_determines_performance](https://huggingface.co/datasets/bethgelab/frequency_determines_performance)

## _Let It Wag!_ Dataset
We provide details about the construction of the Let It Wag! dataset in the [`let_it_wag_datasets`](https://github.com/bethgelab/frequency_determines_performance/tree/main/let_it_wag_datasets) folder. The list of 290 concepts included in the dataset are here: [`let_it_wag_datasets/let_it_wag_class_list.txt`](https://github.com/bethgelab/frequency_determines_performance/blob/main/let_it_wag_datasets/let_it_wag_class_list.txt). The full classification dataset is hosted on Huggingface: [https://huggingface.co/datasets/bethgelab/Let-It-Wag](https://huggingface.co/datasets/bethgelab/Let-It-Wag). All the zero-shot evaluations are released here: [`let_it_wag_datasets/evaluations/let_it_wag_ensemble_results.json`](https://github.com/bethgelab/frequency_determines_performance/blob/main/let_it_wag_datasets/evaluations/let_it_wag_ensemble_results.json)

## Citation
If you find this work useful to your research, please consider citing as:
```
@article{udandarao2024zeroshot,
  title={No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance},
  author={Udandarao, Vishaal and Prabhu, Ameya and Ghosh, Adhiraj and Sharma, Yash and Torr, Philip H. S. and Bibi, Adel and Albanie, Samuel and Bethge, Matthias},
  journal={arXiv preprint arXiv:2404.04125},
  year={2024}
}
```

## Acknowledgements
We thank the authors and contributors of these great papers/repositories for enabling our research: [CLIP](https://github.com/openai/CLIP), [open_clip](https://github.com/mlfoundations/open_clip), [CyCLIP](https://github.com/goel-shashank/CyCLIP), [SLIP](https://github.com/facebookresearch/SLIP/), [NLP-HuJI](https://huggingface.co/nlphuji), [quality-not-quantity](https://github.com/mlfoundations/clip_quality_not_quantity), [RAM++](https://github.com/xinyu1205/recognize-anything), [imagededup](https://github.com/idealo/imagededup), [fastdup](https://github.com/visual-layer/fastdup), [SynthCLIP](https://github.com/hammoudhasan/SynthCLIP), [SuS-X](https://github.com/vishaal27/sus-X/) and [CLIP-OOD](https://github.com/brendel-group/clip-ood).

## Contact
Please feel free to open an issue or email us at [vu214@cam.ac.uk](mailto:vu214@cam.ac.uk) or [ameya@prabhu.be](mailto:ameya@prabhu.be).
