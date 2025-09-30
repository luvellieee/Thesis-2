python src/preprocessing/convert_iob_tags_to_offset_tags.py --output_directory data/ --dataset jjzha/skillspan
# Purpose: Downloads the jjzha/skillspan dataset from Hugging Face and converts its annotations from IOB (Inside-Outside-Beginning) format to offset tags (likely spans of text for skills/knowledge).

python src/preprocessing/prepare_data.py   --annotation_file data/train.jsonl --output_file data/train.json --labels  "[\"SKILL\", \"KNOWLEDGE\"]"
# Purpose: Processes data/train.jsonl (from the previous step) into a format suitable for fine-tuning, filtering for labels “SKILL” and “KNOWLEDGE” (likely for NER-like tasks). Outputs data/train.json.

python src/preprocessing/prepare_data.py   --annotation_file data/validation.jsonl --output_file data/validation.json --labels  "[\"SKILL\", \"KNOWLEDGE\"]"
# Purpose: Same as above, for the validation split (data/validation.json).

python src/preprocessing/prepare_data.py   --annotation_file data/test.jsonl --output_file data/test.json --labels  "[\"SKILL\", \"KNOWLEDGE\"]"
# Purpose: Same, for the test split (data/test.json).