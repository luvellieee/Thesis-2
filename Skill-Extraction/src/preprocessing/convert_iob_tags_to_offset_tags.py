"""
Data post processing, convert data for NER tagging. \
Author:
    Amir Herandi
Date:
    2023/02/20
"""
import os

from argparse import ArgumentParser
import itertools

import html
import pandas as pd
from tqdm import tqdm
import json

import spacy
from spacy.training import biluo_tags_to_spans, iob_to_biluo
from spacy.tokens import Doc

from datasets import load_dataset

DEFAULT_SAVE_UNTAGGED = False


class data_post_processing:
    def __init__(self, resource_directory, output_directory, dataset_name) -> None:
        """
        The data post processing class takes in input of tagging file, and post processes data of different format for NER.

        Parameters
        ----------
        resource_directory: directory containing the config file
        output_directory: directory to save the output files
        dataset_name: name of the dataset
        """
        # get configuration
        if resource_directory is None:
            resource_directory = os.path.dirname(__file__)

        with open(os.path.join(resource_directory, 'config/convert_iob_tags_to_offset_tags.json')) as f:
            config = json.load(f)
        if output_directory is None:
            self.output_directory = config['output_directory']
        else:
            self.output_directory = output_directory
        if dataset_name is None:
            dataset_name = config['dataset']
        self.dataset = load_dataset(dataset_name)

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def convert_iob_tags_to_jsonl(self):
        
        for data_split in self.dataset.keys():
            
            with open(os.path.join(self.output_directory, f'{data_split}.jsonl'), 'w') as f:
                annotation_data = self.dataset[data_split].to_pandas()
                labels = [column.split('_')[-1].upper() for column in annotation_data.columns if 'tags_' in column]
                for index in range(annotation_data.shape[0]):
                    data_entry = annotation_data.iloc[index]
                    tokens = data_entry.tokens
                    tags = {label: [tag if tag == 'O' else f'{tag}-{label}' for tag in data_entry[f'tags_{label.lower()}']] for label in labels}

                    doc = Doc(self.nlp.vocab, words=tokens)
                    json_doc = doc.to_json()
                    text = json_doc['text'].strip()
                    json_tokens = json_doc['tokens']

                    for i, (word, token) in enumerate(zip(tokens, json_tokens)):
                        if i < len(tokens) - 1:
                            token['ws'] = True
                        else:
                            token['ws'] = False
                        token['text'] = word
                        token['end'] = token['end'] - 1

                    biluo_tags = {label: iob_to_biluo(individual_tags) for label, individual_tags in tags.items()}
                    # mismatch can sometimes cause this part to crash (Some cases will still work mostly correctly so
                    # instead of a check we will just use a try except clause)
                    try:
                        spans = list(itertools.chain.from_iterable([biluo_tags_to_spans(doc, individual_tags) for individual_tags in biluo_tags.values()]))

                        json_spans = []
                        for span in spans:
                            json_spans.append({'start': span.start_char, 'end': span.end_char-1, 'label': span.label_,
                                               'token_start': span.start, 'token_end': span.end-1})
                    except IndexError:
                        json_spans = []

                    text = html.unescape(text)

                    prodigy_element = {'text': text, 'tokens': json_tokens, 'spans': json_spans}
                    
                    for split in ['train', 'validation', 'test']:
                        with open(os.path.join(self.output_directory, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                            json.dump(prodigy_element, f, ensure_ascii=False)
                    f.write('\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resource_directory", required=False, default=None, type=str)
    parser.add_argument("--output_directory", required=False, default=None, type=str)
    parser.add_argument("--dataset", required=False, default=None, type=str)
    args = parser.parse_args()

    post_processer = data_post_processing(args.resource_directory, args.output_directory, args.dataset)
    post_processer.convert_iob_tags_to_jsonl()