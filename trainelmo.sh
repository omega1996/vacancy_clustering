#!/usr/bin/env bash

source activate master8_env
export PYTHONPATH="/home/mluser/master8_projects/clustering_vacancies/"
cd /home/mluser/master8_projects/clustering_vacancies/elmo/

nohup python -m elmoformanylangs.biLM train \
    --gpu 0 \
    --train_path /home/mluser/master8_projects/clustering_vacancies/data/corpus/ru_vacancies.raw \
    --config_path configs/cnn_50_100_512_4096_sample.json \
    --model output/ru_vacancies \
    --optimizer adam \
    --lr 0.001 \
    --lr_decay 0.8 \
    --max_epoch 10 \
    --max_sent_len 20 \
    --max_vocab_size 150000 \
    --min_count 3 &

ps -A | grep python