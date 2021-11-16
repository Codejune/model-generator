import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging
from keras.layers import LSTM
from keras.optimizers import Adam

''' Debug configuration '''
DEBUG = False
GPU = True

''' Program configuration '''
info = {
    'name': 'KETI Cloudedge Model Generator',
    'version': '1.0',
}

''' Dataset configuration '''
ROOT_PATH = '/cloudedge'

cluster = ['cluster1', 'cluster2']
DATASET_MINUTE = 20
DATASET_HOUR = DATASET_MINUTE * 60
DATASET_DAY = DATASET_HOUR * 24
DATASET_WEEK = DATASET_DAY * 7
DATASET_MONTH = DATASET_WEEK * 4
DATASET_YEAR = DATASET_MONTH * 12
dataset = {
    # Dataset root path
    'path': [ROOT_PATH + '/cluster1/csv', ROOT_PATH + '/cluster2/csv'],

    'seperate': ',',  # 구분자
    'index_col': ['Time'],
    'parse_dates': ['Time'],
    'infer_datetime_format': True,

    'input_length': DATASET_MINUTE * 5,  # Input dataset length
    'output_length': 1,  # Output dataset length
    'interval': DATASET_MINUTE, # Time interval between input, output
    'stride': 1, # Batch interval

    # 'metrics': ['Power', 'CPU_utilization', 'MEM_utilization', 'NetworkTotalBytes', 'DiskIOWaitTime', 'DiskUsage'],
    # CPU_utilization,MEM_utilization,NetworkTotalBytes,DiskIOMerge,DiskIOWaitTime,DiskUsage
    'metrics': ['CPU_utilization', 'MEM_utilization', 'NetworkTotalBytes', 'DiskUsage'],

}

''' Model configuration '''
TRAIN_FLAG_PATH = ROOT_PATH + '/.train.json'
model = {
    # Model root path
    'path': ROOT_PATH + '/model',

    # Model Layer
    'units': 512,
    'cell': LSTM,
    'bidirectional': True, 
    'layers': 2,
    'dropout': 0.5,
    'optimizer': Adam,
    'learning_rate': 0.001,
    'activation': 'linear',

    # Model metrics
    'loss': 'mean_squared_error',
    'metrics': ['accuracy', 'mean_absolute_error', 'mean_squared_error'],

    # Model training
    'batch_size': 256, 
    'epochs': 100,  # 학습 횟수

    # Model callbacks
    # EarlyStopping
    'min_delta': 0,
    'patience': 10,
    'verbose': 1, 
    'mode': 'auto',
    # ModelCheckpoint
    'save_best_only': True,
}


''' Logger Configuration'''
log = {
    'path': '/var/log/model-generator.log',
    'print_format': '[%(asctime)s.%(msecs)03d: %(levelname).1s %(filename)s:%(lineno)s] %(message)s',
    'time_format': '%Y-%m-%d %H:%M:%S',
}
logger = logging.getLogger()
# Setup logger level 
logger.setLevel(logging.INFO)
# Setup logger format
formatter = logging.Formatter(fmt=log['print_format'], datefmt=log['time_format'])
# Setup logger handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(log['path'])
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)