import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
import csv
import random
from utils import data_reader
from utils import config
import os
import matplotlib.pyplot as plt

LABELS = {
    "negative": -1,
    "neutral": 0,
    "positive": 1
}


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


class ClassificationModel:
    def __init__(self, bert_model=config.bert_model, gpu=False, seed=0):

        self.gpu = gpu
        self.bert_model = bert_model

        self.train_df = data_reader.load_train_dataset(config.data_path)
        self.val_df = data_reader.load_dev_dataset(config.data_path)
        self.test_df = data_reader.load_test_dataset(config.data_path)

        self.num_classes = len(LABELS)

        self.model = None
        self.optimizer = None
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        # to plot loss during training process
        self.plt_x = []
        self.plt_y = []

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.gpu:
            torch.cuda.manual_seed_all(seed)

    def __init_model(self):
        if self.gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        print(torch.cuda.memory_allocated(self.device))
        # log available cuda
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    def new_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, num_labels=self.num_classes)
        self.__init_model()

    def load_model(self, path_model, path_config):
        self.model = BertForSequenceClassification(BertConfig(path_config), num_labels=self.num_classes)
        self.model.load_state_dict(torch.load(path_model))
        self.__init_model()

    def save_model(self, path_model, path_config, acc, f1):

        model_save_path = os.path.join(path_model,
                                       'model_{}_{:.4f}_{:.4f}_{:.4f}'.format(iter, acc, f1))

        torch.save(self.model.state_dict(), model_save_path)

        if not os.path.exists(path_config):
            os.makedirs(path_config)
        with open(path_config, 'w') as f:
            f.write(self.model.config.to_json_string())

    def train(self, epochs, batch_size=config.batch_size, lr=config.lr, plot_path=None , model_path=None, config_path=None):

        model_params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1,
                                  t_total=int(len(self.train_df) / batch_size) * epochs)

        nb_tr_steps = 0
        train_features = data_reader.convert_examples_to_features(self.train_df, config.MAX_SEQ_LENGTH, self.tokenizer)

        # create tensor of all features
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        # class weighting
        _, counts = np.unique(self.train_df['label'], return_counts=True)
        class_weights = [sum(counts) / c for c in counts]
        # assign wight to each input sample
        example_weights = [class_weights[e] for e in self.train_df['label']]
        sampler = WeightedRandomSampler(example_weights, len(self.train_df['label']))
        train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=batch_size)

        self.model.train()
        for e in range(epochs):
            print(f"Epoch {e}")
            f1, acc = self.val()
            print(f"\nF1 score: {f1}, Accuracy: {acc}")
            if model_path is not None and config_path is not None:
                self.save_model(model_path, config_path, acc, f1)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                loss.backward()

                if plot_path is not None :
                    self.plt_y.append(loss.item())
                    self.plt_x.append(nb_tr_steps)
                    self.save_plot(plot_path)

                nb_tr_steps += 1
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.gpu:
                    torch.cuda.empty_cache()

    def val(self, batch_size=config.batch_size):
        eval_features = data_reader.convert_examples_to_features(self.val_df, config.MAX_SEQ_LENGTH, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        f1, acc = 0, 0
        nb_eval_examples = 0

        for input_ids, input_mask, segment_ids, gnd_labels in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            predicted_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
            acc += np.sum(predicted_labels == gnd_labels.numpy())
            tmp_eval_f1 = f1_score(predicted_labels, gnd_labels, average='macro')
            f1 += tmp_eval_f1 * input_ids.size(0)
            nb_eval_examples += input_ids.size(0)

        return f1 / nb_eval_examples, acc / nb_eval_examples

    def save_plot(self, path):

        fig, ax = plt.subplots()
        ax.plot(self.plt_x, self.plt_y)

        ax.set(xlabel='Training steps', ylabel='Loss')

        fig.savefig(path)
        plt.close()

    def create_test_predictions(self, path):
        tests_features = data_reader.convert_examples_to_features(self.x_test, [-1] * len(self.test_df),
                                                                 config.MAX_SEQ_LENGTH,
                                                                 self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in tests_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in tests_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in tests_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in tests_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=16)

        predictions = []
        inverse_labels = {v: k for k, v in LABELS}

        for input_ids, input_mask, segment_ids, gnd_labels in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                encoded_layers, logits = self.model(input_ids, segment_ids, input_mask)

            predictions += [inverse_labels[p] for p in list(np.argmax(logits.detach().cpu().numpy(), axis=1))]
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for i, prediction in enumerate(predictions):
                writer.writerow([int(self.x_test_ids[i]), prediction])

        return predictions

