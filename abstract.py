import abc
import torch

class AbstractCFRecommender(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def model_to_cuda(self):
        pass

    @abc.abstractmethod
    def predict(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def encoder(self):
        raise NotImplementedError

    @abc.abstractmethod
    def encoder_predict(self):
        pass

    
    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
    
    def load(self,load_path):
        self.load_state_dict(torch.load(load_path))
    
    @abc.abstractmethod
    def loss(self, data_batch)->dict:
        raise NotImplementedError
