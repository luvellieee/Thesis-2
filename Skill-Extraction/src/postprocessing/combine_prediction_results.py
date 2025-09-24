


"""
Data post processing, convert data for NER tagging. \
Author:
    Amir Herandi
Date:
    2023/02/21
"""
import html
import os

import argparse
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm
import json

from spacy.training import biluo_tags_to_spans, iob_to_biluo
import spacy
from spacy.tokens import Doc

import re
from fuzzywuzzy import fuzz


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * (len(words) - 1)
        spaces.append(False)
        return Doc(self.vocab, words=words, spaces=spaces)


class Evaluation:
    def __init__(self, resource_directory, base_file, input_file, prediction_file, output_file, tokenizer) -> None:
        """
        Combine evaluation data with base data for analysis.

        Parameters
        ----------
        resource_directory: directory containing the config file
        """
        # get configuration
        if resource_directory is None:
            resource_directory = os.path.dirname(__file__)

        with open(os.path.join(resource_directory, 'config/combine_prediction_results.json')) as f:
            config = json.load(f)
        if input_file is None:
            input_file = config['input_file']
        if base_file is None:
            self.base_file = config['base_file']
        else:
            self.base_file = base_file
        if prediction_file is None:
            self.prediction_file = config['prediction_file']
        else:
            self.prediction_file = prediction_file
        if output_file is None:
            self.output_file = config['output_file']
        else:
            self.output_file = output_file
        if tokenizer is None:
            self.tokenizer = config['tokenizer']
        else:
            self.tokenizer = tokenizer

        self.annotation_data = pd.read_json(path_or_buf=input_file)

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # White space tokenizer should be used for the SkillSPAN dataset for consistency
        if self.tokenizer == 'white_space':
            self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)

    def find_errors_eval(self, input_list):
        try:
            eval(input_list)
            return False
        except SyntaxError:
            return True

    def find_best_match(self, sentence, subphrase, subphrase_with_context):
        try:
            # Compile the regular expression pattern
            pattern = re.compile(r'\b' + re.escape(subphrase) + r'\b', re.IGNORECASE)

            # Find all matches of the subphrase in the sentence
            matches = [(match.start(), match.end()) for match in re.finditer(pattern, sentence)]
            assert len(matches) > 0
        except AssertionError:  # with some skill spans we need to do this instead
            # Compile the regular expression pattern
            pattern = re.compile(re.escape(subphrase), re.IGNORECASE)

            # Find all matches of the subphrase in the sentence
            matches = [(match.start(), match.end()) for match in re.finditer(pattern, sentence)]


        if not matches:
            return None, None, None

        best_match = None
        best_score = -1
        best_start = 0
        best_end = 0

        # Iterate over each match
        for start, end in matches:
            # Calculate the context range
            context_start = max(0, start - (len(subphrase_with_context) - len(subphrase)))
            context_end = min(len(sentence), end + (len(subphrase_with_context) - len(subphrase)))

            # Extract the context around the match
            context = sentence[context_start:context_end]

            # Perform fuzzy matching with the context
            score = fuzz.ratio(context, subphrase_with_context)

            # Update the best match if necessary
            if score > best_score:
                best_match = context
                best_score = score
                best_start = start
                best_end = end

        return best_match, best_start, best_end

    def parse_label_and_context_list(self):
        predictions = pd.read_json(self.prediction_file, lines=True)
        base_data = pd.read_json(self.base_file, lines=True)[['text']]
        self.annotation_data = pd.concat([self.annotation_data, predictions], axis=1)
        base_data = base_data.drop_duplicates(subset=['text'])
        # Remove the leading and trailing **'s
        self.annotation_data['text'] = self.annotation_data['input'].apply(lambda x: x[3:-3] if x is not None else None)
        self.annotation_data = self.annotation_data.merge(base_data, how='left', on=['text'])
        self.annotation_data['predict_error'] = self.annotation_data['predict'].apply(self.find_errors_eval)
        error_data = self.annotation_data[self.annotation_data['predict_error']]
        error_data.to_csv(f'{os.path.dirname(self.output_file)}/error_data.csv', index=False)
        print(f'{len(error_data)} errors found.')
        self.annotation_data['predict'] = self.annotation_data.apply(lambda x: '[]' if x['predict_error'] else x['predict'], axis=1)
        self.annotation_data['label'], self.annotation_data['predict'] = zip(
            *self.annotation_data.apply(lambda x: [eval(x['label']), eval(x['predict'])], axis=1))
        with open(self.output_file, mode='w') as f:
            for i, row in tqdm(self.annotation_data.iterrows(), total=len(self.annotation_data)):
                clean_text = row['text']
                prediction = row['predict']

                clean_doc = self.nlp(clean_text)

                tokens = [{'id': token.i, 'start': token.idx, 'end': token.idx + len(token.text) - 1,
                           'ws': token.whitespace_ == ' ', 'text': token.text} for token in clean_doc]

                skill_spans = []
                for skill_type in prediction:
                    skill_list = prediction[skill_type]
                    for skill in skill_list:
                        skill_span = skill['skill_span']
                        if 'context' in skill:
                            context = skill['context']
                        else:
                            context = skill['skill_span']

                        best_match, best_start, best_end = self.find_best_match(clean_text, skill_span, context)
                        if best_match is not None:
                            span = clean_doc.char_span(best_start, best_end)
                            if span is not None:
                                token_start = span[0].i
                                token_end = span[-1].i
                                skill_spans.append(
                                    {'start': best_start, 'end': best_end-1, 'label': skill_type,
                                     'token_start': token_start, 'token_end': token_end}
                                )
                            else:
                                skill_spans.append(
                                    {'start': best_start, 'end': best_end-1, 'label': skill_type}
                                )

                json_doc  = {'text': clean_text, 'tokens': tokens, 'spans': skill_spans}

                json.dump(json_doc, f, ensure_ascii=False)
                f.write('\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resource_directory", required=False, default=None, type=str)
    parser.add_argument("--base_file", required=False, default=None, type=str)
    parser.add_argument("--input_file", required=False, default=None, type=str)
    parser.add_argument("--prediction_file", required=False, default=None, type=str)
    parser.add_argument("--output_file", required=False, default=None, type=str)
    parser.add_argument("--tokenizer", required=False, default=None, type=str)
    args = parser.parse_args()
    evaluator = Evaluation(args.resource_directory, args.base_file, args.input_file, args.prediction_file, args.output_file, args.tokenizer)
    evaluator.parse_label_and_context_list()