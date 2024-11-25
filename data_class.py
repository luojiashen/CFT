from dataclasses import dataclass
from matplotlib.figure import Figure
import pprint
import os

from cft_setting import model_parameters
from recorder import MAX_STAGNANT_TIMES

@dataclass
class Setting:
    model_name:str
    data_name:str
    basic_params:dict
    model_params:dict

    def get_params(self)->dict:
        return {**self.basic_params, **self.model_params}
      
    def __eq__(self, other) -> bool:
        if not other:
            return False
        
        if self.model_name == other.model_name and self.data_name == other.data_name:
            return self.Dict_Eq(self.model_params, other.model_params)\
                and self.Dict_Eq(self.basic_params, other.basic_params)
        else:
            return False
    
    def __contains__(self, kv_tuple):
        k, v = kv_tuple[0], kv_tuple[1]
        # 基础训练参数
        if k in self.basic_params and str(self.basic_params[k]) == str(v):
            return True
        # 模型参数
        if k in self.model_params and str(self.model_params[k]) == str(v):
            return True
        return False
    
    def __str__(self) -> str:
        return f"{self.model_name} on {self.data_name}"
    
    def show(self):
        print("*"*30,"basic settings","*"*30)
        pprint.pprint(self.basic_params)
        print("*"*30,"model settings","*"*30)
        pprint.pprint(self.model_params)
        
    @staticmethod
    def Dict_Eq(d1:dict, d2:dict)->bool:
        for k, v in d1.items():
            if k not in d2:
                return False
            if v != d2[k]:
                return False
        return True
        
    @staticmethod
    def Dict_Hash(d:dict)->int:
        return sum([hash(k)+hash(v) if type(v) is not list else hash(v[0]) 
                    for k, v in d. items()])
    
    def __hash__(self) -> int:
        return hash(self.model_name)+hash(self.data_name) \
                + self.Dict_Hash(self.basic_params)\
                + self.Dict_Hash(self.model_params)


class SettingFactory:
    @staticmethod
    def Get_Setting(setting_dict:dict) -> Setting:
        # 获取模型参数名
        model_p = model_parameters

        model_name = setting_dict['model_name']
        data_name = setting_dict['data_name']
        basic_params, model_params = dict(), dict()
        # 基础参数和模型参数分为两个字典
        for k, v in setting_dict.items():
            if k in model_p:
                model_params[k] = v
            else:
                basic_params[k] = v
        if "sep_line" in basic_params:
            basic_params.pop("sep_line")
        setting = Setting(model_name, data_name, basic_params, model_params)
        return setting

@dataclass
class Record:
    losses:dict
    training_performance:dict
    training_time_cost:float

    def get_best_performance(self)->dict:
        '''获取模型性能字典
        '''
        best_p_d = dict()
        for k, v in self.training_performance.items():
            best_p_d[k] = max(v)
        return best_p_d
    
    def __hash__(self) -> int:
        loss_hash = Setting.Dict_Hash(self.losses)
        training_per_hash = Setting.Dict_Hash(self.training_performance)
        return loss_hash + training_per_hash + hash(self.training_time_cost)

    def is_legal(self):
        '''record是否合法，即最优值的位置是否满足早停机制
        '''
        performance_list = list(self.training_performance.values())[0]
        
        if len(performance_list)>MAX_STAGNANT_TIMES and max(performance_list) == performance_list[-MAX_STAGNANT_TIMES]: # 早停机制为10
            return True
        return False
    
    def max_train_performance(self)->float:
        return max(list(self.training_performance.values())[0])
    
    def __lt__(self, other):
        if self.max_train_performance() < other.max_train_performance():
            return True
        return False
    
    def plot_to_figure(self, fig:Figure)->Figure:
        losses = self.losses
        performance = self.training_performance
        axes0 = fig.add_subplot(1, 2, 1)
        for k, v in losses.items():
            # 对损失进行归一化，方便显示
            v_max, v_min = max(v), min(v)
            v = [(vi-v_min)/(v_max-v_min) for vi in v]
            axes0.plot(v, label=k)
        axes0.grid()
        axes0.legend()
        axes0.set_title("训练损失(normalized)")

        axes1 = fig.add_subplot(1, 2, 2)
        for k, v in performance.items():
            axes1.plot(v, label=f"{k}:{v[-1]:.4f}/{max(v):.4f}")
        axes1.grid()
        axes1.legend()
        axes1.set_title("推荐指标")

        return fig

class RecordFactory:
    @staticmethod
    def Get_Record(rec:dict):
        return Record(rec['losses'], 
                      rec['training_performance'],
                      rec["training_time_cost"])

@dataclass
class Set_Rec:
    timestamp:str
    setting:Setting
    record:Record

    @property
    def model_path(self)->str:
        return os.path.join("results","trained_model", self.timestamp+".pt")
        
    def __hash__(self) -> int:
        # if self.setting.data_name == 'frappe' and self.setting.model_name == "dnsgcl":
        #      if self.setting.model_params["neg_c"] == 50 and self.setting.model_params["cl_lambda"] == 0.0:
        #         self.setting.show()
        #         print(hash(self.setting))
        return hash(self.setting)

    def __eq__(self, value) -> bool:
        return self.setting == value.setting
    
    def __str__(self) -> str:
        return f"Result of {self.setting}"
    
    def __gt__(self, other):
        if self.record > other.record:
            return True
        return False
    
    def __contains__(self, kv_tuple):
        if kv_tuple in self.setting:
            return True
        return False


    

    


