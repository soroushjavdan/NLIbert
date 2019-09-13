import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/dialogue_nli_extra")
parser.add_argument("--save_path", type=str, default='save/')
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--plot_path", type=str, default='save/plot')
parser.add_argument("--bert_model", type=str, default='bert-base-cased')
parser.add_argument("--gpu", action='store_true')
parser.add_argument("--load_frompretrain", type=str, default="None")
parser.add_argument("--model_state_path", type=str, default="None")
parser.add_argument("--model_config_path", type=str, default="None")

arg = parser.parse_args()
print(arg)

TRAIN_FILE = 'dialogue_nli_EXTRA_uu_train.json'
TEST_FILE = 'dialogue_nli_EXTRA_uu_test.json'
DEV_FILE = 'dialogue_nli_EXTRA_uu_dev.json'
MAX_SEQ_LENGTH = 210

data_path = arg.data_path
save_path = arg.save_path
lr = arg.lr
batch_size = arg.batch_size
plot_path = arg.plot_path
bert_model = arg.bert_model
epochs = arg.epochs
USE_GPU = arg.gpu
load_frompretrain = arg.load_frompretrain
model_config_path = arg.model_config_path
model_state_path = arg.model_state_path
