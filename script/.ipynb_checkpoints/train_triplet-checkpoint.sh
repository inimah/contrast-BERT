#!/bin/bash
source MINICONDA_PATH
conda activate simcse-env

python src/train.py \
    --model_name_or_path LazarusNLP/simcse-indobert-base \
    --method supervised-triplets-filtered \
    --eval_attribute title-fact \
    --datapath HUGGINGFACE-triplet-path \
    --evalpath HUGGINGFACE-pairwise-path \
    --output_dir OUTPUT-DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 160 \
    --evaluation_strategy steps \
    --eval_transfer \
    --metric_for_best_model PosPairsCLS \
    --load_best_model_at_end \
    --eval_steps 20 \
    --save_steps 20 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"