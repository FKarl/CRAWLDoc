#!/usr/bin/env bash
echo "Running robustness check"
echo "========================"
for doi in "10.1007" "10.1016" "10.1109" "10.1145" "10.3390" "10.48550"
do
    echo "Running for DOI: $doi"
    python3 eval_ranking.py --run_name=test+robustness_check+$doi --max_context=2048  --test_data=dataset/test_$doi.json --query_model=./models/robustness_check+10.1007/query_model --document_model=./models/robustness_check+10.1007/document_model --tokenizer=./models/robustness_check+10.1007/document_model
done