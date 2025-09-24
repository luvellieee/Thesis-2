"""
Data preparation process for preparing fine-tuning data for skill extraction. \
Author:
    Kevin Liu
    Amir Herandi
Date:
    2023/02/07
"""
import json
import os
import random
from argparse import ArgumentParser
import html

import pandas as pd
from tqdm import tqdm

INSTRUCTION_PROMPT = [
    "You are a helpful information extraction system. Your job is to extract skill entities and knowledge entities from the given sentence.",
    "Your job is to extract skills and knowledge from given text.",
]


class data_preparation:
    def __init__(self, resource_directory, annotation_file, output_file, labels) -> None:
        """
        The data preparation class takes in input of annotation file, and prepare data of different format for fine-tuning purpose.

        Parameters
        ----------
        resource_directory: directory containing the config file
        annotation_file: input jsonl file
        output_file: output json file used by llama factory
        labels: list of labels
        """
        # get configuration
        if resource_directory is None:
            resource_directory = os.path.dirname(__file__)

        with open(os.path.join(resource_directory, 'config/prepare_data.json')) as f:
            config = json.load(f)
        if annotation_file is None:
            annotation_file = config['annotation_file']
        if output_file is None:
            self.output_file = config['output_file']
        else:
            self.output_file = output_file
        if labels is None:
            self.labels = config['labels']
        else:
            self.labels = labels
        self.labels = eval(self.labels)
        self.annotation_data = pd.read_json(path_or_buf=annotation_file, lines=True)
        self.instruction_prompt = INSTRUCTION_PROMPT

    def label_and_context_list_data(self):
        """
        This function generate data of "label_and_context_list" format
        Example output:
            [
                {
                    "instruction": "You are a helpful information extraction system. Your job is to extract skill entities and knowledge entities from the given sentence.",
                    "input": "** You will be working in an end- to-end cross-functional team being responsible for implementing and promoting all QA relevant topics on team level . **",
                    "{\"SKILL\": [{\"skill_span\": \"implementing and promoting all QA relevant topics\", "context": \"for implementing and promoting all QA relevant topics on\"}], \"KNOWLEDGE\": [{\"skill_span\": \"QA\", \"context\": \"all QA relevant\"}]}"
                },
            ]
        """
        prompt = []
        for index in tqdm(range(self.annotation_data.shape[0])):
            data_entry = self.annotation_data.iloc[index]
            text, spans, tokens = data_entry.text, data_entry.spans, data_entry.tokens
            prompt_entry = {
                "instruction": random.choice(self.instruction_prompt),
                "input": "** " + text + " **",
            }
            response_entry = {label: [] for label in self.labels}
            if (isinstance(spans, float)) or (spans is None):
                if pd.isnull(spans):
                    spans = []
            for span in spans:
                span_type = span["label"]
                token_start_index, token_end_index = span["token_start"], span["token_end"]
                span_tokens = tokens[token_start_index:token_end_index+1]
                context = ""
                skill_span = ""
                if token_start_index == 0:
                    context += "** "
                else:
                    context += tokens[token_start_index-1]['text']
                    if tokens[token_start_index-1]['ws']:
                        context += " "
                for i, token in enumerate(span_tokens):
                    skill_span += token['text']
                    context += token['text']
                    if token['ws']:
                        if i < len(span_tokens)-1:
                            skill_span += " "
                        context += " "
                if token_end_index >= len(tokens) - 1:
                    context += " **"
                else:
                    context += tokens[token_end_index+1]['text']
                skill_span = html.unescape(skill_span)
                context = html.unescape(context)
                response_entry[span_type].append({"skill_span": skill_span, "context": context})
            prompt_entry["output"] = json.dumps(response_entry, ensure_ascii=False)
            prompt.append(prompt_entry)
        self.write_file(prompt, self.output_file)

    def write_file(self, prompt, output_file):
        """
        Write prompt object into a .json file
        """
        prompt = json.dumps(prompt, indent=2, ensure_ascii=False)
        with open(output_file, "w") as json_file:
            json_file.write(prompt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resource_directory", required=False, default=None, type=str)
    parser.add_argument("--annotation_file", required=False, default=None, type=str)
    parser.add_argument("--output_file", required=False, default=None, type=str)
    parser.add_argument("--labels", required=False, default=None, type=str)
    args = parser.parse_args()

    data_processer = data_preparation(args.resource_directory, args.annotation_file, args.output_file, args.labels)
    data_processer.label_and_context_list_data()
