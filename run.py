from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from RAGGNN import RAGGNN
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model

config = Config(model=RAGGNN, dataset='diginetica', config_file_list=['RAGGNN.yaml'])
init_seed(config['seed'], config['reproducibility'])
init_logger(config)
logger = getLogger()
logger.info(config)
dataset = create_dataset(config)
logger.info(dataset)
train_data, valid_data, test_data = data_preparation(config, dataset)
model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
logger.info(model)
trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
test_result = trainer.evaluate(test_data, show_progress=True)
logger.info('best valid result: {}'.format(best_valid_result))
logger.info('test result: {}'.format(test_result))
