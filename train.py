import argparse
from settings import MODEL_SETTING_MAP
import pprint
from itertools import product
import copy
from torch_geometric import seed_everything

from trainer import ModelDataParameterFactor, Trainer
from settings import BasicSettings
from results import Results, Setting, SettingFactory
import numpy as np

def dict_type(s):
    '''str(param=value) to dict'''
    s = [x.split("=") for x in s.split(' ')]
    return dict(s)

def get_args():
    parser = argparse.ArgumentParser(description="training model")
    # mf_bpr, mf_dns
    parser.add_argument('--model_name', type = str, default='cft')
    # frappe, ml_1m, gowalla
    parser.add_argument('--data_name', type = str, default='gowalla')
    # model's parameters
    parser.add_argument("--grid_search", type=int, help="hyperparameter search or not", 
                        default = 0)
    args = parser.parse_args()
    model_name, data_name = args.model_name, args.data_name

    return model_name, data_name, args.grid_search

def generate_param_list(params:BasicSettings):
    search_space = params.Get_Search_Space()
    if not search_space:
        raise NotImplemented("没有设置搜索空间")
    # dict to tuple
    search_tuples = []
    for k, vs in search_space.items():
        search_tuples.append([(k, v) for v in vs])
    # 搜索空间展开
    params_list = [] 
    search_tuples = list(product(*search_tuples))
    for param_tuple in search_tuples:
        temp_params = copy.copy(params)
        for k, v in param_tuple:
            setattr(temp_params, k, v)
        params_list.append(temp_params)

    return params_list, search_tuples

if __name__ == "__main__":
    seed_everything(2024)
    model_name, data_name, grid_search = get_args()
    if not grid_search:
        print(f"**************************单模型训练****************************")
        # 获取parameters
        params = MODEL_SETTING_MAP[model_name](data_name)
        # 查询是否存在参数相同的模型（无论是否训练完）
        params_set:Setting = SettingFactory.Get_Setting(params.__dict__)
        ts = Results.setting_exists(params_set)
        if ts:
            print("there is a saved model")
            # 加载模型

        pprint.pprint(params.__dict__, compact=True)
        model, data = ModelDataParameterFactor.get_model_data_by_params(params)
        Trainer(params).Train_BP(model, data)
    else:# 超参搜索
        # 获取parameters
        params_origin = MODEL_SETTING_MAP[model_name](data_name)
        params_list, param_tuple = generate_param_list(params_origin)
        pprint.pprint(param_tuple)
        for i, params in enumerate(params_list):
            print(f"**************************超参搜索第{i+1}/{len(params_list)}轮{param_tuple[i]}****************************")
            pprint.pprint(params.__dict__)
            model, data = ModelDataParameterFactor.get_model_data_by_params(params)
            Trainer(params).Train_BP(model, data)




    