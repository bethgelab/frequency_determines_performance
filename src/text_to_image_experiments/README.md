# T2I Experiments: Data Collection, Preprocessing and Image Generation

### Save JSON files from HEIM
We use the results for T2I models provided by [HEIM](https://crfm.stanford.edu/helm/heim/latest/). To see the models we report, please refer to our paper.

Run `save_json.py` to save JSON files from HEIM. Use the following command:
```
cd t2i
python save_json.py --save_root <data_root_directory> --dataset <dataset>
```
where `<data_root_directory>` is the root directory to save the HEIM data and `<datset>` denote one of the datasets we currently support for json extraction from HEIM. This command will save the prompts and metric scores for each model.

The datasets we currently support are: `cub200`,`daily_dalle`,`detection`,`draw_bench`,`mscoco`,`parti_prompts`,`relational_understanding`,`winoground`

Remember to use the exact dataset name as mentioned here when running the command.

### Data Preprocessing
#### Dataset-Specific Preprocessing
Once you have saved the JSON files, you can preprocess the result for each dataset using the following command:
```
cd t2i
python score.py --root <data_root_directory> --save_dir <results_directory> --dataset <dataset>
```
where `<data_root_directory>` is the root directory where the json files from HEIM are saved, `<results_directory>` is the directory where you wish to store the result files and `<datset>` denote one of the datasets we currently support for json extraction from HEIM.

With this command, you will have access to dataset and model-specific results in .pkl files and dataset-specific results for all models in .json files.

#### Aggregating Results
To aggregate the concept results for all models and all datasets, use the following command:
```
cd t2i
python score.py --save_dir <results_directory> --combine
```
This command will save the aggregated results across all datasets for a specific evaluation metric. 
We currently provide results for the following metrics:
- Expected Aesthetics Score
- Expected CLIP Score
- Expected LPIPS Score
- Expected PSNR Score
- Expected SSIM Score
- Expected UIQI Score
- Expected Human Aesthetics Score
- Expected Human Image-Text Alignment Score
- Max Aesthetics Score
- Max CLIP Score

### Image Generation
We also provide code to generate images for the classes in the `Let It Wag!` dataset. Given that the dataset is split into `common` and `fine_grained` categories, the prompts are also split into two .pkl files in `let_it_wag_datasets` directory, namely `let_it_wag_common_prompts_for_image_gen.pkl` and `let_it_wag_fine_grained_prompts_for_image_gen.pkl`. 

To generate images, use the following command:
```
cd t2i
python image_generation.py --load_prompts <path_to_prompts_file> --save_dir <directory_to_save_images> --pipe <t2i-pipeline> --model_id <huggingface_model_id>
```

where `<path_to_prompts_file>` is the path to one of the two .pkl files mentioned above. Refer to `image_generation.py` for more details on the remaining arguments. Feel free to experiment with different T2I models, batch sizes (for faster generations) and image sizes.
We currently provide code for [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [SDv2](https://huggingface.co/stabilityai/stable-diffusion-2-1) and [Dreamlike-Photoreal](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0).

### Human Evaluation Experiment on Public Figures
To do a more controlled study on concept frequencies and image generation models, we ran an additional experiment where we collected frequency data of public figures. All the scripts for this experiment are provided in the [human_experiment_evaluation](https://github.com/bethgelab/frequency_determines_performance/tree/main/src/text_to_image_experiments/human_experiment_evaluation) folder. For more details, please see Appendix C in the paper.

