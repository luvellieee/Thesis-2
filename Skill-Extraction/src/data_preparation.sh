python src/preprocessing/convert_iob_tags_to_offset_tags.py --output_directory data/ --dataset jjzha/skillspan

python src/preprocessing/prepare_data.py   --annotation_file data/train.jsonl --output_file data/train.json --labels  "[\"SKILL\", \"KNOWLEDGE\"]"

python src/preprocessing/prepare_data.py   --annotation_file data/validation.jsonl --output_file data/validation.json --labels  "[\"SKILL\", \"KNOWLEDGE\"]"

python src/preprocessing/prepare_data.py   --annotation_file data/test.jsonl --output_file data/test.json --labels  "[\"SKILL\", \"KNOWLEDGE\"]"