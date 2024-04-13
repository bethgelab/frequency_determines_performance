# Setting up datasets

## Pretraining datasets
We provide the download scripts for each of the pretraining datasets we used: CC-3M, CC-12M, YFCC-15M, LAION-400M, LAION-Aesthetics, SynthCI-30M, in their respective folders. For the [Mayilvahanan et al. experiment](https://arxiv.org/abs/2310.09562), we have released the exact sample indices from LAION400M that are included in the final dataset [here](https://huggingface.co/datasets/bethgelab/frequency_determines_performance/resolve/main/paths_leave_out_near_val_150m_whole_data_new_pruning_method.npy) with the authors' permission---we thank the authors again for their great work!

## Downstream datasets
For the zero-shot classification experiments, please follow the data download setup from the [SuS-X github repository](https://github.com/vishaal27/SuS-X/blob/main/data/DATA.md). For the retrieval experiments, we directly use the splits provided on huggingface: [flickr1k](https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval) and [coco](https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval). For text-to-image generation experiments, we use the datasets from [HEIM](https://crfm.stanford.edu/helm/heim/latest/)---for more details on these evaluations, please see the `src/text_to_image_experiments` folder.
