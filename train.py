import pprint
import argparse


from torch_geometric import seed_everything

from cft_setting import CFTSettings
from trainer import ModelDataParameterFactor, Trainer
from data_class import Setting, SettingFactory


def get_args():
    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument('--model_name', type = str, default='cft')
    parser.add_argument('--data_name', type = str, default='ml_100k')
    args = parser.parse_args()
    model_name, data_name = args.model_name, args.data_name
    return model_name, data_name


if __name__ == "__main__":
    seed_everything(2024)
    model_name, data_name = get_args()
    params = CFTSettings(data_name)
    params_set:Setting = SettingFactory.Get_Setting(params.__dict__)
    pprint.pprint(params.__dict__, compact=True)
    model, data = ModelDataParameterFactor.get_model_data_by_params(params)
    Trainer(params).Train_BP(model, data)





    