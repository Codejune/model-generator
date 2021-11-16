import os
import glob
import static
import config
import threading
from config import logger
from model import build


class BuildThread(threading.Thread):
    """스레드 클래스
    각 Node, Pod별로 해당 스레드 오브젝트가 1개씩 할당되며, 이 스레드에서 model/build() 함수가 호출된다
    """
    def __init__(self, param):
        threading.Thread.__init__(self)
        self.name = param.name
        self.is_builded = build(param)
        self.daemon = True

    def result(self):
        return self.is_builded


class Cluster:
    def __init__(self, name, dataset_path):
        """
        Initialization cluster object and setup node, pod object list
        
        path: Dataset directory path
        """
        # Cluster name
        self.name = name
        # Cluster dataset_path
        self.dataset_path = dataset_path

        # Node list
        self.nodes = list()
        for node_dataset_path in glob.glob(os.path.join(self.dataset_path, '*.csv')):
            self.nodes.append(Node(self.name, node_dataset_path))

    def __repr__(self):
        out = dict()
        for i in self.nodes:
            out[i.name] = i.pods
        return out.__repr__()

    def build_model(self):
        """
        Start prediction model thread
        """
        for node in self.nodes:
            node.build_model()
            for pod in node.pods:
                pod.build_model()

    def update_flag(self):
        """학습 주기 플래그 업데이트
        """
        flag = dict()
        for node in self.nodes:
            if node.name in static.flag:
                flag[node.name] = static.flag[node.name]
            for pod in node.pods:
                if pod.name in static.flag:
                    flag[pod.name] = static.flag[pod.name]
        static.flag = flag


class Node(Cluster):
    def __init__(self, cluster, dataset_path):
        """
        Initialization cluster object and setup node, pod object list
        
        path: Dataset directory path
        """
        # Node name
        self.name = os.path.basename(dataset_path).split('.')[0]
        # Node location
        self.location = cluster  # 클라우드 엣지의 경우 노드 위치 확인 필요
        # Node dataset path
        self.dataset_path = dataset_path

        # Node prediction model path
        path = os.path.join(
            config.ROOT_PATH, self.location, 'model', self.name)
        if not os.path.isdir(path):
            logger.info('%s model directory does not exist, create' % (path))
            os.makedirs(path)
        self.model_path = os.path.join(
            config.ROOT_PATH, self.location, 'model', self.name + '.h5')
        self.pods = list()
        for pod_dataset_path in glob.glob(os.path.join(os.path.dirname(self.dataset_path), self.name, '*.csv')):
            self.pods.append(Pod(self.location, self.name, pod_dataset_path))

    def __repr__(self):
        """Print 찍었을 때 출력 결과 호출
        """
        out = dict()
        out[self.name] = list()
        for i in self.pods:
            out[self.name].append(i.name)
        return out.__repr__()

    def build_model(self):
        """
        Start prediction model thread
        """
        # Setup thread
        self.thread = BuildThread(self)
        self.thread.daemon = False
        self.thread.start()
        self.thread.join()


class Pod(Node):

    def __init__(self, cluster, node, dataset_path):
        # Pod name
        self.name = os.path.basename(dataset_path).split('.')[0]
        # Pod location
        self.location = node
        self.cluster = cluster
        # Pod dataset path
        self.dataset_path = dataset_path
        # Pod prediction model path
        self.model_path = os.path.join(
            config.ROOT_PATH, self.cluster, 'model', self.location, self.name + '.h5')

    def __repr__(self):
        """Print 찍었을 때 출력 결과 호출
        """
        return self.name

    def build_model(self):
        """
        Start prediction model thread
        """
        if 'kube' in self.name:
            return
        # Setup thread
        self.thread = BuildThread(self)
        self.thread.start()
        self.thread.join()
