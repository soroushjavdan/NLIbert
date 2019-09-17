import os
import json
from utils import config


def repair_train_data():
    path = './data/dialogue_nli_extra/dialogue_nli_EXTRA_uu_train.jsonl'
    if os.path.exists(path):
        with open(path) as f:
            txt = f.readline()
            t = txt[:-44]
            t = t + ']'
            n = json.dumps(t)
            json_obj = json.loads(n)
            with open('./data/dialogue_nli_extra/'+config.REPARIED_TRAIN_FILE, 'w') as fout:
                json.dump(json_obj, fout)
    else:
        raise ValueError('palce the training data in /data/dialogue_nli_extra/')

repair_train_data()