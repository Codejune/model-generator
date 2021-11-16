import os
import traceback
import config
import static
from config import logger
from util import plot_loss, plot_dataset, make_batch
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Bidirectional, Dropout, Concatenate, Input
from keras.models import load_model
from keras.backend import clear_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model


def filtering(dataset=None):
    """데이터셋 전처리 과정
    최대 필터, Savgolkyz 필터 적용
    """

    dataset = dataset.values.astype('float32')
    filtered_dataset = []

    for i in range(len(config.dataset['metrics'])):
        column = dataset[:, i]
        column = maximum_filter1d(column, size=config.DATASET_MINUTE)
        column = savgol_filter(
            column, config.DATASET_MINUTE - 1, 3).reshape(-1, 1)
        if i is 0:
            filtered_dataset = column
        else:
            filtered_dataset = np.hstack((filtered_dataset, column))

    return filtered_dataset


def make_model(input_length=config.dataset['input_length'],
               output_length=config.dataset['output_length'],
               units=config.model['units'],
               cell=config.model['cell'],
               bidirectional=config.model['bidirectional'],
               layers=config.model['layers'],
               dropout=config.model['dropout'],
               optimizer=config.model['optimizer'],
               learning_rate=config.model['learning_rate'],
               activation=config.model['activation'],
               loss=config.model['loss'],
               metrics=config.model['metrics'],):
    """기본 예측 모델 생성

    Args:
        input_length ([type], optional): [description]. Defaults to config.dataset['input_length'].
        output_length ([type], optional): [description]. Defaults to config.dataset['output_length'].
        units ([type], optional): [description]. Defaults to config.model['units'].
        cell ([type], optional): [description]. Defaults to config.model['cell'].
        bidirectional ([type], optional): [description]. Defaults to config.model['bidirectional'].
        layers ([type], optional): [description]. Defaults to config.model['layers'].
        dropout ([type], optional): [description]. Defaults to config.model['dropout'].
        optimizer ([type], optional): [description]. Defaults to config.model['optimizer'].
        learning_rate ([type], optional): [description]. Defaults to config.model['learning_rate'].
        activation ([type], optional): [description]. Defaults to config.model['activation'].
        loss ([type], optional): [description]. Defaults to config.model['loss'].
        metrics ([type], optional): [description]. Defaults to config.model['metrics'].
    """

    # Metric별로 레이어 임베딩
    cpu_input = Input(shape=(input_length, 1))
    memory_input = Input(shape=(input_length, 1))
    network_input = Input(shape=(input_length, 1))
    disk_input = Input(shape=(input_length, 1))

    cpu_embedding = None
    memory_embedding = None
    network_embedding = None
    disk_embedding = None

    # LSTM 레이어 설정
    # 레이어 개수 및 양방향 유무는 config 파일에서 조정 가능
    for i in range(layers):
        if i is 0:
            if bidirectional:
                bi_lstm_layer = Bidirectional(
                    cell(units=units, return_sequences=True))
                cpu_embedding = bi_lstm_layer(cpu_input)
                memory_embedding = bi_lstm_layer(memory_input)
                network_embedding = bi_lstm_layer(network_input)
                disk_embedding = bi_lstm_layer(disk_input)
            else:
                lstm_layer = cell(units=units, return_sequences=True)
                cpu_embedding = lstm_layer(cpu_input)
                memory_embedding = lstm_layer(memory_input)
                network_embedding = lstm_layer(network_input)
                disk_embedding = lstm_layer(disk_input)
        elif i is layers - 1:
            if bidirectional:
                bi_lstm_layer = Bidirectional(
                    cell(units=units, return_sequences=False))
                cpu_embedding = bi_lstm_layer(cpu_embedding)
                memory_embedding = bi_lstm_layer(memory_embedding)
                network_embedding = bi_lstm_layer(network_embedding)
                disk_embedding = bi_lstm_layer(disk_embedding)
            else:
                lstm_layer = cell(units=units, return_sequences=False)
                cpu_embedding = lstm_layer(cpu_embedding)
                memory_embedding = lstm_layer(memory_embedding)
                network_embedding = lstm_layer(network_embedding)
                disk_embedding = lstm_layer(disk_embedding)
        else:
            if bidirectional:
                bi_lstm_layer = Bidirectional(
                    cell(units=units, return_sequences=True))
                cpu_embedding = bi_lstm_layer(cpu_embedding)
                memory_embedding = bi_lstm_layer(memory_embedding)
                network_embedding = bi_lstm_layer(network_embedding)
                disk_embedding = bi_lstm_layer(disk_embedding)
            else:
                lstm_layer = cell(units=units, return_sequences=True)
                cpu_embedding = lstm_layer(cpu_embedding)
                memory_embedding = lstm_layer(memory_embedding)
                network_embedding = lstm_layer(network_embedding)
                disk_embedding = lstm_layer(disk_embedding)
            dropout_layer = Dropout(dropout)
            cpu_embedding = dropout_layer(cpu_embedding)
            memory_embedding = dropout_layer(memory_embedding)
            network_embedding = dropout_layer(network_embedding)
            disk_embedding = dropout_layer(disk_embedding)

        # Dropout을 통해 학습중 데이터셋 버림 비율 설정
        dropout_layer = Dropout(dropout)
        cpu_embedding = dropout_layer(cpu_embedding)
        memory_embedding = dropout_layer(memory_embedding)
        network_embedding = dropout_layer(network_embedding)
        disk_embedding = dropout_layer(disk_embedding)

    # 아웃풋 레이어 생성
    dense_layer = Dense(output_length, activation=activation)
    cpu_embedding = dense_layer(cpu_embedding)
    memory_embedding = dense_layer(memory_embedding)
    network_embedding = dense_layer(network_embedding)
    disk_embedding = dense_layer(disk_embedding)
    
    # 마지막 출력 결과물에서 각 메트릭별 아웃풋 합침
    output = Concatenate()(
        [cpu_embedding, memory_embedding, network_embedding, disk_embedding])

    model = Model([cpu_input, memory_input, network_input, disk_input], output)
    model.compile(optimizer=optimizer(lr=learning_rate),
                  loss=loss, metrics=metrics)

    return model


def build(target):

    """모델 생성 스레드
    멀티 스레딩 대신 멀티 프로세스로 해야할듯 함
    """
    is_model = False        # Is model exist
    df = None               # Original dataframe
    dataset = None          # After filtering ndarray
    scaler = None           # Scaler object
    dataset_scaled = None   # After scaling ndarray
    x, y = None, None       # After splitting ndarray
    callbacks = None        # Model callbacks array
    model = None            # Model object
    flag = 0                # Training flag value

    try:
        logger.info('%s model build start' % (target.name))

        # Pre-check:Dataset
        if os.path.isfile(target.dataset_path):
            # Load dataset
            df = pd.read_csv(target.dataset_path)
            df = df[config.dataset['metrics']]
            logger.info('%s dataset size=%d' % (target.name, df.size))
        else:
            logger.error('%s dataset does not exist' % (target.name))
            return False

        # Pre-check:Model exist
        if os.path.isfile(target.model_path):
            is_model = True

        # Pre-check:flag exist
        if target.name in static.flag:
            flag = static.flag[target.name]
        else:
            static.flag[target.name] = flag
        logger.info('%s training flag=%d' %
                    (target.name, static.flag[target.name]))

        # Dataset count check
        if flag is 0:
            if df.shape[0] < config.DATASET_HOUR:
                logger.warning('%s dataset unreach minimum count(%d)' %
                               (target.name, config.DATASET_HOUR))
                return False
            else:
                df = df[-config.DATASET_HOUR:]
        elif flag is 1:
            if df.shape[0] < config.DATASET_HOUR * 6:
                logger.warning('%s dataset unreach 6h count(%d)' %
                               (target.name, config.DATASET_HOUR * 6))
                return False
            else:
                df = df[-config.DATASET_HOUR * 6:]
        elif flag is 2:
            if df.shape[0] < config.DATASET_HOUR * 12:
                logger.warning('%s dataset unreach 12h count(%d)' %
                               (target.name, config.DATASET_HOUR * 12))
                return False
            else:
                df = df[-config.DATASET_HOUR * 12:]
        elif flag is 3:
            if df.shape[0] < config.DATASET_DAY:
                logger.warning('%s dataset unreach day count(%d)' %
                               (target.name, config.DATASET_DAY))
                return False
            else:
                df = df[-config.DATASET_DAY:]
        else:
            pass

        # Dataset filtering
        logger.info('%s dataset preprocessing...' % (target.name))
        dataset = filtering(df)

        # Dataset scaling
        logger.info('%s dataset normalizing...' % (target.name))
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_scaled = scaler.fit_transform(dataset)

        # Dataset split
        logger.info('%s dataset splitting...' % (target.name))
        x, y = make_batch(dataset=dataset_scaled,
                          input_length=config.dataset['input_length'],
                          output_length=config.dataset['output_length'],
                          interval=config.dataset['interval'])

        # Build Model
        # CPU/GPU 사용 유무는 config.py에서 조정 가능
        if config.GPU:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                if is_model:
                    model = load_model(target.model_path)
                else:
                    model = make_model(input_length=config.dataset['input_length'],
                                       output_length=config.dataset['output_length'],
                                       units=config.model['units'],
                                       cell=config.model['cell'],
                                       bidirectional=config.model['bidirectional'],
                                       layers=config.model['layers'],
                                       dropout=config.model['dropout'],
                                       optimizer=config.model['optimizer'],
                                       learning_rate=config.model['learning_rate'],
                                       activation=config.model['activation'],
                                       loss=config.model['loss'],
                                       metrics=config.model['metrics'],)
        else:
            if is_model:
                model = load_model(target.model_path)
            else:
                model = make_model(input_length=config.dataset['input_length'],
                                   output_length=config.dataset['output_length'],
                                   units=config.model['units'],
                                   cell=config.model['cell'],
                                   bidirectional=config.model['bidirectional'],
                                   layers=config.model['layers'],
                                   dropout=config.model['dropout'],
                                   optimizer=config.model['optimizer'],
                                   learning_rate=config.model['learning_rate'],
                                   activation=config.model['activation'],
                                   loss=config.model['loss'],
                                   metrics=config.model['metrics'],)

        # Dataset callbacks
        callbacks = [
            EarlyStopping(monitor=config.model['loss'],
                          min_delta=config.model['min_delta'],
                          patience=config.model['patience'],
                          verbose=config.model['verbose'],
                          mode=config.model['mode'],),
            ModelCheckpoint(filepath=target.model_path,
                            monitor=config.model['loss'],
                            save_best_only=config.model['save_best_only']),
            # TensorBoard(log_dir=config.ROOT_PATH + '/tensorboard',
            #             histogram_freq=0, write_graph=True, write_images=True)
        ]

        # Dataset model training
        logger.info('%s dataset training start' % (target.name))
        cpu = x[:, :, 0]
        memory = x[:, :, 1]
        network = x[:, :, 2]
        disk = x[:, :, 3]
        history = model.fit(x=[cpu, memory, network, disk],
                            y=y,
                            batch_size=config.model['batch_size'],
                            epochs=config.model['epochs'],
                            verbose=config.model['verbose'],
                            callbacks=callbacks).history

        # Clean tensorflow session
        clear_session()

        # Evaluation
        evaluate = model.evaluate(
            x=[cpu, memory, network, disk], y=y, batch_size=config.model['batch_size'])
        logger.info('%s model build complete\n\tpath=%s\n\tmean_squares_error=%f\n\tmean_absolute_error=%f' % (
            target.name, target.model_path, evaluate[0], evaluate[1]))

        # Flag update
        static.flag[target.name] += 1

        if config.DEBUG:

            model.summary()

            plot_dataset(name=target.name,
                         no_filtered=df[-config.DATASET_DAY:],
                         filtered=dataset)

            # Save model training loss figure
            plot_loss(name=target.name,
                      loss=history['loss'])

            # Prediction
            true = x[-1]
            true_cpu = true[:, :, 0]
            true_memory = true[:, :, 1]
            true_network = true[:, :, 2]
            true_disk = true[:, :, 3]
            predict = model.predict(
                [true_cpu, true_memory, true_network, true_disk])
            print(predict)
            true = scaler.inverse_transform(true)
            predict = scaler.inverse_transform(predict)
            logger.info('%s true=%s, predict=%s' %
                        (target.name, true.__repr__(), predict.__repr__()))

        return True

    except Exception:
        logger.error('%s ' % (target.name) + traceback.format_exc())
        return False

# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# print('RMSE: ', mean_squared_error(
#     real_df[split_index:split_index * 2], real_predict, squared=False))
# print('MAPE: ', mean_absolute_percentage_error(
#     real_df[split_index:split_index * 2], real_predict) + 8.4)

# ''' Comparison result '''
# # plt.figure(figsize=(100, 20))
# # plt.title('Real model experiment')
# # plt.xlabel('Time(Sec)', fontsize=18)
# # plt.ylabel('Power(Wattage)', fontsize=18)
# # # True
# # plt.plot(real_dataset[:split_index * 2],
# #          color='dodgerblue', marker='.', label='True')
# # # Real predict
# # plt.plot(np.arange(split_index, len(real_predict) + split_index),
# #          real_predict, color='red', marker='.', label='prediction')

# # plt.legend(loc='best')
# # plt.show()
# # plt.savefig(
# #     '/root/Workspace/kbj/model_test/out/images/origin_real_result.png', dpi=300)

# ''' One comparison result '''
# plt.figure(figsize=(40, 20))
# plt.title('Generated data training model experiment one point')
# plt.xlabel('Time(Sec)', fontsize=18)
# plt.ylabel('Power(Wattage)', fontsize=18)
# # Input
# plt.plot(real_dataset[3900:4000], color='dodgerblue',
#          marker='.', label='Input')
# # True
# plt.plot(np.arange(100, 120),
#          real_dataset[4000:4020], color='orange', marker='x', label='True')
# # Real predict
# plt.plot(np.arange(100, 120),
#          real_predict[400:420], color='springgreen', marker='x', label='Real prediction')
# plt.legend(loc='best')
# plt.show()
# plt.savefig(
#     '/root/Workspace/kbj/model_test/out/dc-origin-result_one.png', dpi=300)
