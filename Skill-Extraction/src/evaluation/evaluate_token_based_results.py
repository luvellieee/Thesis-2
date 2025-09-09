"""
Evaluation pipeline for LLaMa Factory skill extraction model. \
Author:
    Amir
Date:
    2023/02/22
"""

import json
import pandas as pd

from argparse import ArgumentParser

from nervaluate import Evaluator


class evaluator:
    def __init__(self, tags=['SKILL', 'KNOWLEDGE'], loader='default') -> None:
        self.tags = tags
        self.loader = loader

    def _metric(self, token_label_list, predicted_token_label_list):
        evaluator = Evaluator(token_label_list, predicted_token_label_list, tags=self.tags, loader=self.loader)
        self.pred = evaluator.pred
        results, evaluation_agg_entities_type, evaluation_indices, evaluation_agg_indices = evaluator.evaluate()
        return results, evaluation_agg_entities_type, evaluation_indices, evaluation_agg_indices


def build_eval_pipeline_nervaluate(prediction_file, label_file, tags=['SKILL', 'KNOWLEDGE']):
    evaluator_ = evaluator(tags=tags)
    results_df = pd.read_json(prediction_file, lines=True)
    results_df['spans'] = results_df['spans'].apply(lambda x: x if not x is None else [])

    label_df = pd.read_json(label_file, lines=True)
    label_df['spans'] = label_df['spans'].apply(lambda x: x if not x is None else [])
    label_df = label_df.reset_index(drop=True)
    token_label_list = label_df['spans'].values

    predicted_token_label_list = results_df['spans'].values

    print(f'Total data: {len(predicted_token_label_list)}')

    print(f'Total label data: {len(token_label_list)}')

    results, evaluation_agg_entities_type, evaluation_indices, evaluation_agg_indices = evaluator_._metric(
        token_label_list, predicted_token_label_list)

    return results['strict'], {entity: entity_metric['strict'] for entity, entity_metric in evaluation_agg_entities_type.items()}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prediction_file", required=True, type=str)
    parser.add_argument("--label_file", required=True, type=str)
    args = parser.parse_args()
    results, evaluation_agg_entities_type = build_eval_pipeline_nervaluate(
        args.prediction_file, args.label_file, tags=['SKILL', 'KNOWLEDGE'])
    print(f'results score:\n{json.dumps(results, indent=2)}')
    print(f'results_per_tag score:\n{json.dumps(evaluation_agg_entities_type, indent=2)}')
