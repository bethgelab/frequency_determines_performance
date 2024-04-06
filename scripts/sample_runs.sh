# image-search rampp model run
# cambridge cluster:
python image_search_run_rampp.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --chunk_idx 0
# galvani cluster:
python image_search_run_rampp.py --cache_dir /mnt/qb/work/bethge/bkr405/huggingface_cache --features_dir /mnt/qb/work/bethge/bkr405/data/blind_name_only_transfer/features --results_dir /mnt/qb/work/bethge/bkr405/data/blind_name_only_transfer/results --chunk_idx 0

# misalignment degree quantification:
# cambridge cluster:
python misalignment_degree_quantification.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --pt_dataset cc3m

# image-search inverted index creation:
# cambridge cluster:
python image_search_inverted_index_creation.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --pt_dataset cc3m
# galvani cluster:
# python image_search_inverted_index_creation.py --cache_dir /mnt/qb/work/bethge/bkr405/huggingface_cache --features_dir /mnt/qb/work/bethge/bkr405/data/blind_name_only_transfer/features --results_dir /mnt/qb/work/bethge/bkr405/data/blind_name_only_transfer/results --pt_dataset laion400m
python image_search_inverted_index_creation.py --cache_dir /mnt/qb/work/bethge/bkr405/huggingface_cache --features_dir /mnt/qb/work/bethge/bkr405/data/blind_name_only_transfer/features --results_dir /mnt/qb/bethge/bkr405/blind_name_only_transfer/results --pt_dataset laion400m

# integrated-search inverted index
# cambridge cluster
python integrated_search_matches_inverted_index.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --pt_dataset cc3m --downstream_dataset cifar10

# image-search inverted index
# cambridge cluster
python image_search_matches_inverted_index.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --pt_dataset cc3m --threshold 0.5 --downstream_dataset cifar10
# galvani cluster
python image_search_matches_inverted_index.py --cache_dir /mnt/qb/work/bethge/bkr405/huggingface_cache --features_dir /mnt/qb/work/bethge/bkr405/data/blind_name_only_transfer/features --results_dir /mnt/qb/work/bethge/bkr405/data/blind_name_only_transfer/results --pt_dataset laion400m --threshold 0.5 --downstream_dataset cifar10

# text-search inverted index
python text_search_matches_inverted_index.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --downstream_dataset cifar10 --search_method lemmatized --pt_dataset laion400m --do_chunked_search True
# python text_search_matches_inverted_index.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --downstream_dataset t2icoco --search_method lemmatized --pt_dataset laion_aesthetics --do_chunked_search True

# text-search dask
python text_search_matches_dask.py --pt_dataset cc3m --downstream_dataset cifar10

# zero-shot results
python zero_shot_eval.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --backbone RN50 --pretraining cc3m --text_prompts simple --dataset let_it_wag_common

# split text search index
python split_text_inverted_index.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --num_shards 10

# alignment-prob get-coarse-grained-mapping
python alignment_probability_get_coarse_grained_mapping.py --dataset cifar10 --mapping_dataset LVIS

# image-text retrieval eval
python retrieval_eval.py --cache_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/pretrained_networks/clip --results_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/results --features_dir /home/vu214/rds/rds-t2-cs151-lSmP1cwRttU/vu214/blind-name-only-transfer-resources/features --dataset coco --backbone RN50 --pretraining cc3m