#!/usr/bin/env bash
set -e
python src/train_lstm.py --csv data/sample_tweets.csv --models_dir models --config configs/config.yaml
