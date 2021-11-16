import matplotlib.pyplot as plt
import numpy as np
import config
from config import logger


def make_batch(dataset=None,
               input_length=config.dataset['input_length'],
               output_length=config.dataset['output_length'],
               interval=config.dataset['interval'],
               stride=config.dataset['stride']):
    """데이터셋을 모델이 인식/학습할 수 있는 Window/Batch로 변환

    Args:
        dataset ([type], optional): 원본 데이터셋. Defaults to None.
        input_length ([type], optional): 입력 길이. Defaults to config.dataset['input_length'].
        output_length ([type], optional): 출력 길이. Defaults to config.dataset['output_length'].
        interval ([type], optional): 입력과 출력 사이 간격. Defaults to config.dataset['interval'].
        stride ([type], optional): 윈도우당 간격을 얼마나 설정할 것인지. Defaults to config.dataset['stride'].
    """

    x = []
    y = []

    interval = interval - 1

    for i in range(0, len(dataset) - input_length - output_length - interval, stride):
        x.append(dataset[i:i + input_length])
        y.append(dataset[i + input_length + interval:i +
                         input_length + interval + output_length])

    return np.array(x), np.array(y)


def plot_dataset(name,
                 no_filtered,
                 filtered):
    """데이터셋 결과 그래프 생성
    """
    plt.figure(figsize=(60, 20))
    plt.title('%s power dataset' % (name), fontsize=18)
    plt.xlabel('Time(Sec)', fontsize=18)
    plt.ylabel('Power(Wattage)', fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.plot(no_filtered, color='dodgerblue',
             marker='.', label='No filtered')
    plt.plot(filtered, color='red', marker='.', label='Filterd')
    plt.legend(loc='best')
    plt.savefig('/cloudedge/images/%s_dataset.png' %
                (name), dpi=100)
    plt.close()


def plot_loss(name,
              loss):
    """학습 손실 결과 출력
    """
    plt.figure(figsize=(60, 20))
    plt.title('%s training loss' % name)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.plot(loss,
             label='loss',
             color='dodgerblue')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('/cloudedge/images/%s_loss.png' % (name), dpi=100)
    plt.close()
