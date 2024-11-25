import abc 
import json

class BasicSettings(metaclass = abc.ABCMeta):
    def __init__(self, model_name, data_name):
        self.batch_size:int = 4096
        self.lr:float = 1e-3
        self.epoches:int = 1500
        self.test_epoch = 5
        self.topk = 40
        self.device = "cuda:0"
        self.model_name = model_name
        self.data_name = data_name

    
    def Save(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.__dict__, f, indent = 2)





