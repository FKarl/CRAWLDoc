#!/usr/bin/env bash
echo "Running robustness check"
echo "========================"
for doi in "10.1007" "10.1016" "10.1109" "10.1145" "10.3390" "10.48550"
do
    echo "Running for DOI: $doi"
    python3 train_retrieval.py --learning_rate=3e-05 --num_accumulation_steps=32 --run_name=robustness_check+$doi --max_context=2048 --epochs=16 --train_data=dataset/train_all_except_$doi.json --val_data=dataset/empty.json --disable_early_stopping
done