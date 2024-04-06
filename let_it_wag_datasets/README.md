# Documentation for cleaning the Let-It-Wag datasets

We follow these steps to ensure thorough cleaning of the datasets.
- We download from three different web-sources to ensure a high diversity of images. We source our datasets from Flickr (used by most image datasets of the past decade), DuckDuckGo and Bing-Search. To ensure no overlap with the pre-training datasets of VLMs, we make sure that all the images we download are uploaded to the web post January, 2023.
- For cleaning, we first apply an InceptionNet pre-trained on ImageNet to do a very minor outlier removal by setting very high thresholds for the outlier detection (0.9 for common and 0.95 for fine-grained classes)
- We then manually clean the left-over images per class and human-verify them.
- We then take the human label-verified images and de-duplicate them per class with perceptual hashing with a bit-threshold of 10 (`deduplicate_let_it_wag_datasets.py`)
- Then, we class-balance the remaining images to give our final test datasets (`class_balance_let_it_wag_datasets.py`).

You can find all the evaluation results (Tab. 6 in the paper) in `evaluations/let_it_wag_ensemble_results.json`.