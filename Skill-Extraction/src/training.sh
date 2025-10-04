python -c 'from llamafactory.train.tuner import run_exp; run_exp()' Skill-Extraction/src/train/config/skill_span_train.yaml
# This starts teaching Llama-3-8B-Instruct (a smart program) using a settings file (skill_span_train.yaml).
# This line downloads meta-llama/Llama-3-8B-Instruct from a website called Hugging Face

python -c 'from llamafactory.train.tuner import run_exp; run_exp()' Skill-Extraction/src/train/config/skill_span_evaluate.yaml
# This checks how well the model finds skills using another settings file (skill_span_evaluate.yaml).

python Skill-Extraction/src/postprocessing/combine_prediction_results.py --base_file Skill-Extraction/data/test.jsonl  --input_file Skill-Extraction/data/test.json --prediction_file saves/Meta-Llama-3-8B-Instruct/lora/results/generated_predictions.jsonl --output_file saves/Meta-Llama-3-8B-Instruct/lora/results/processed_predictions.jsonl --tokenizer white_space
# This organizes the model’s answers into a file.

python Skill-Extraction/src/evaluation/evaluate_token_based_results.py --prediction_file saves/Meta-Llama-3-8B-Instruct/lora/results/processed_predictions.jsonl --label_file Skill-Extraction/data/test.jsonl
# This checks if the model’s answers are correct.