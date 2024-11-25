from os import path, makedirs
import json
from time import time
from collections import defaultdict

MAX_STAGNANT_TIMES = 10

class Recorder:
    def __init__(self):
        self.losses_records = defaultdict(list) # 存储损失
        self.performance_records = defaultdict(list) # 存储模型性能
        self.best_performance:float = 0.0

        self.total_time = 0
        self.last_time = time()

        self._max_stagnant_times = MAX_STAGNANT_TIMES
        self.stagnant_count = 0
    
    def show_best_performance(self):
        print("[Best of all]:", end = " ")
        for k, v in self.performance_records.items():
            if len(v) > 0:
                print(f" {k}: {max(v):.7f} ", end = " ")

    def Save(self, save_path):
        training_record = {"losses":self.losses_records,
                           "training_performance":self.performance_records,
                           "training_time_cost":self.total_time}
        with open(save_path, 'w') as f:
            json.dump(training_record, f, indent = 2)
        
    def Time_Update(self)->float:
        time_cost = time() - self.last_time
        self.total_time += time_cost
        self.last_time = time()
        return time_cost

    def Should_Early_Stop(self, eval_dict:dict)->int:
        eval_criteria = list(eval_dict.values())[0]
        if eval_criteria > self.best_performance:
            self.stagnant_count = 0
            self.best_performance = eval_criteria
            return 0
        else:
            self.stagnant_count += 1
            if self.stagnant_count > self._max_stagnant_times:
                return 1
            else:
                return -1

    @staticmethod
    def Dict_Show(d:dict):
        print("|".join([f"{k}:{v:.7f}" for k,v in d.items()]))

    def Record_Loss(self, loss_dict:dict):
        for loss_name, loss_value in loss_dict.items():
            self.losses_records[loss_name].append(loss_value)
    
    def Record_Performance(self, eval_dict:dict):
        for eval_name, eval_value in eval_dict.items():
            self.performance_records[eval_name].append(eval_value)

    def Epoch_Record(self, epoch:int, loss_dict:dict):
        loss_dict = dict(zip(loss_dict.keys(), [float(v.cpu().detach().numpy()) for v in loss_dict.values()]))
        
        self.Epoch_Print(epoch)
        self.Dict_Show(loss_dict)
        self.Record_Loss(loss_dict)

    def Epoch_Print(self, epoch:int):
        print(f"Epoch:{epoch}|Time Cost:{self.Time_Update():.4f}s", end = "|")

    def Performance_Record(self, epoch:int, eval_dict:dict)->int:
        self.Epoch_Print(epoch)
        self.Dict_Show(eval_dict)
        self.Record_Performance(eval_dict)
        return self.Should_Early_Stop(eval_dict)