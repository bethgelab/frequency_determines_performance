## Running the RAM++ model

For running the RAM++ model, we provide the `image_search_run_rampp.py` script. This script loads a particular tar shard from the pre-training datasets, and runs the RAM++ model on the shard directly without having to untar the shard files. This helps with storage efficiency. The `tardataset.py` script implements the dataloader from the tar shards.

The script takes in the following params:
- `pt_dataset`: The pretraining dataset to run the analysis (one of cc3m, cc12m, yfcc15m, laion_aesthetics, laion400m).
- `load_path`: The path where the pretraining dataset's tar files are located.
- `chunk_idx`: The index of the tar file to process
- `class_jsons`: Path to json file containing the gpt4-generated descriptions of all the concepts
- `batch_size_rampp`: Batch-size for running the RAM++ model
- `confidence_threshold`: The confidence threshold to apply on the output of the RAM++ model. This threshold determines at which probability value we consider an image to be tagged with a particular concept.
- `pretrained`: Path to pretrained RAM++ model checkpoint.
- `image_size`: Input image size to RAM++ model
- `cache_dir`: Cache folder to save/load CLIP model weights
- `features_dir`: Folder to save the encoded features
- `results_dir`: Folder to save the results of the experiment
