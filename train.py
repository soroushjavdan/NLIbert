import bert_classifier
from utils import config
from utils import data_repairer
from utils.data_reader import load_train_dataset

if config.need_repair:
    data_repairer.repair_train_data()

cm = bert_classifier.ClassificationModel(gpu=False, seed=0)
if config.load_frompretrain is not None:
    cm.load_model(config.model_state_path, config.model_config_path)
else:
    cm.new_model()

# cm.save_model(config.save_path + '/model',config.save_path + '/config')

cm.train(config.epochs, config.batch_size, config.lr, config.plot_path, config.save_path + 'model',
         config.save_path + 'config')

cm.create_test_predictions("./pred.csv")


if __name__ == '__main__':
    print('running')
    load_train_dataset()