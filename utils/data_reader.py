import numpy as np
import pandas as pd
import os
from utils import config


class BertInputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def load_test_dataset(path):
    if not os.path.exists(path + config.TEST_FILE):
        raise ValueError('DataSet path does not exist ')
        return

    data = pd.read_json(path + config.TEST_FILE)
    # reform labels
    data['label'] = data['label'].map(
        lambda label: 0 if label == 'negative' else (1 if label == 'neutral' else 2))

    return data


def load_dev_dataset(path):
    if not os.path.exists(path + config.DEV_FILE):
        raise ValueError('DataSet path does not exist ')
        return
    data = pd.read_json(path + config.DEV_FILE)
    # reform labels
    data['label'] = data['label'].map(
        lambda label: 0 if label == 'negative' else (1 if label == 'neutral' else 2))

    return data


def load_train_dataset(path):
    if not os.path.exists(path + config.REPARIED_TRAIN_FILE):
        raise ValueError('DataSet path does not exist ')
        return

    data = pd.read_json(path + config.REPARIED_TRAIN_FILE)
    # reform labels
    data['label'] = data['label'].map(
        lambda label: 0 if label == 'negative' else (1 if label == 'neutral' else 2))

    return data


def convert_examples_to_features(pandas, max_seq_length, tokenizer):
    features = []

    for i, r in pandas.iterrows():
        first_tokens = tokenizer.tokenize(r['sentence1'])
        sec_tokens = tokenizer.tokenize(r['sentence2'])
        tokens = ["[CLS]"] + first_tokens + ["[SEP]"] + sec_tokens
        if len(sec_tokens) + len(first_tokens) > max_seq_length - 1:
            tokens = tokens[:(max_seq_length - 1)]
        tokens = tokens + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        features.append(
            BertInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=r['label']))
    return features
