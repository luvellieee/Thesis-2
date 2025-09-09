python -c 'from llamafactory.train.tuner import run_exp; run_exp()' src/train/config/skill_span_train.yaml

python -c 'from llamafactory.train.tuner import run_exp; run_exp()' src/train/config/skill_span_evaluate.yaml

python src/postprocessing/combine_prediction_results.py --base_file data/test.jsonl  --input_file data/test.json --prediction_file saves/Meta-Llama-3-8B-Instruct/lora/results/generated_predictions.jsonl --output_file saves/Meta-Llama-3-8B-Instruct/lora/results/processed_predictions.jsonl --tokenizer white_space

python src/evaluation/evaluate_token_based_results.py --prediction_file saves/Meta-Llama-3-8B-Instruct/lora/results/processed_predictions.jsonl --label_file data/test.jsonl