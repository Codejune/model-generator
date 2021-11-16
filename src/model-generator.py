import os
import json
import traceback
import static
from time import sleep
import config
from config import logger
from component import Cluster


def load_flag():
    """플래그 로드
    """
    with open(file=config.TRAIN_FLAG_PATH, mode='r', encoding='utf-8') as file:
        static.flag = json.load(file)


def save_flag(flag=None):
    """플래그 저장
    """
    # Create empty json file
    with open(file=config.TRAIN_FLAG_PATH, mode='w', encoding='utf-8') as file:
        json.dump(obj=flag, fp=file, indent="\t")


def init():
    """
    Initilization function
    """

    logger.info('Initializing...')
    try:

        """ TODO 디렉토리 자동 검색화 시켜야함 """
        # Check dataset directory
        # while True:
        #     if not (os.path.isdir(config.dataset['path'])):
        #         logger.warning('Dataset(%s) directory doesn\'t exist, waiting for 60 seconds' % (config.dataset['path']))
        #         sleep(60)
        #     else:
        #         break

        # Check training flag file
        if os.path.isfile(config.TRAIN_FLAG_PATH):
            load_flag()
        else:
            static.flag = dict()
            save_flag(static.flag)
    except Exception:
        logger.error(traceback.format_exc())


def main():
    logger.info(
        'Starting ' + config.info['name'] + ' version ' + config.info['version'])
    static.cluster1 = Cluster(config.cluster[0], config.dataset['path'][0])
    logger.info('Cluster1 information: %s' % (static.cluster1))
    static.cluster2 = Cluster(config.cluster[1], config.dataset['path'][1])
    logger.info('Cluster2 information: %s' % (static.cluster2))

    while True:
        static.cluster1.build_model()
        static.cluster2.build_model()

        static.cluster1.update_flag()
        static.cluster2.update_flag()
        save_flag(static.flag)
        
        logger.info('Waiting for 1 hour to next build model')
        sleep(1200)


if __name__ == '__main__':
    init()
    main()
