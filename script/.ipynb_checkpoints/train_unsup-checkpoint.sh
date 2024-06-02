#!/bin/bash
source MINICONDA_PATH
conda activate simcse-env

python train.py \
    --model_name_or_path LazarusNLP/simcse-indobert-base \
    --output_dir OUTPUT-DIR  \
    --method unsupervised \
    --attribute title \
    --eval_attribute title-fact \
    --datapath HUGGINGFACE-unsup-title-path \
    --evalpath HUGGINGFACE-pairwise-path \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 160 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --metric_for_best_model PosPairsCLS \
    --load_best_model_at_end \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"