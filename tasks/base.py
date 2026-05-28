from abc import ABC, abstractmethod


class Task(ABC):

    @abstractmethod
    def make_model(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def sample_batch(self, batch_size: int):
        raise NotImplementedError

    def full_batch(self):
        """返回全量训练数据，用于 full batch 训练。"""
        return self.x_train, self.y_train

    @abstractmethod
    def loss(self, model, batch):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model):
        raise NotImplementedError
