import os
from time import ctime
from torch.optim import Adam

from cft import CFT
from data_cft import DataCFT
from basic_settings import BasicSettings
from recorder import Recorder
from evaluator import Evaluator


class Trainer:
    def __init__(self, params) -> None:
        self.params = params
        self.recorder = Recorder()
        self.Path_Init()
    
    def Root_Valid(self, root):
        '''若目录root不存在则创建root目录'''
        if not os.path.exists(root):
            os.makedirs(root)
        
    def Path_Init(self):
        '''实验过程中需要存储数据的路径初始化'''
        root = "results"
        model_root = os.path.join(root, "trained_model")
        record_root = os.path.join(root, "records")
        setting_root = os.path.join(root, "settings")
        self.Root_Valid(model_root)
        self.Root_Valid(record_root)
        self.Root_Valid(setting_root)

        timestamp = ctime().replace(':',"-")
        
        self.model_save_path = os.path.join(model_root, timestamp+".pt")
        self.record_save_path = os.path.join(record_root, 
                                             timestamp+".json")
        self.setting_save_path = os.path.join(setting_root, 
                                             timestamp+".json")
    
    def Get_Train_Loader(self, data):
        if self.params.model_name in ("dns", 'ssm', 'cft'):
            return data.Get_Train_Loader(self.params.batch_size, self.params.neg_c)
        
        return data.Get_Train_Loader(self.params.batch_size)

    def Train_BP(self, model, 
                 data):
        
        params = self.params
        optimizer = Adam(model.parameters(), lr = params.lr)

        trainloader = self.Get_Train_Loader(data)
        testlabels = data.Get_Test_Labels()
        uid_rated_items = data.Get_User_Rated_Items()
        
        # train
        model.model_to_cuda()
        for epoch in range(params.epoches):
            model.train()
            for data_batch in trainloader:
                data_batch = [i.to(device = params.device) for i in data_batch]
                optimizer.zero_grad()
                loss_dict = model.loss(data_batch)
                loss = list(loss_dict.values())[-1]
                loss.backward()
                optimizer.step()
            self.recorder.Epoch_Record(epoch, loss_dict)
            # print("训练记录保存")
            self.recorder.Save(self.record_save_path)
            if epoch%params.test_epoch == 0:
                trainloader.dataset.shuffle()
                eval_dict = Evaluator.Training_Evaluate(model.predict(params.topk, uid_rated_items), testlabels)
                earlystop = self.recorder.Performance_Record(epoch, eval_dict)
                if earlystop == 1:
                    print("Parameter Save | Training Record Save")
                    params.Save(self.setting_save_path)
                    self.recorder.Save(self.record_save_path)
                    self.recorder.show_best_performance()
                    print("Early Stop")
                    break
                elif earlystop == -1:
                    print("Stagnanting")
                else:
                    print("[Performance Improved]")
                    model.save(self.model_save_path)
                    params.Save(self.setting_save_path)
                    self.recorder.Save(self.record_save_path)
    
    @staticmethod
    def model_eval(model, params, data):
        uid_rated_items =  data.Get_User_Rated_Items()
        testlabels = data.Get_Test_Labels()
        eval_dict = Evaluator().Training_Evaluate(model.predict(params.topk, 
                                                                uid_rated_items), 
                                                 testlabels)
        print(f"{params.model_name}-{params.data_name} performance", eval_dict)
        
class ModelDataParameterFactor:
    def __init__(self):
        pass

    @staticmethod
    def get_model_params(data, 
                         params):
        if params.model_name in ['cft']:
            return (data.user_num, data.item_num, 
                    data.sse(params.sse_dim, params.sse_type), 
                    data.pathes_ptypes(params.walk_len, params.path_num, 
                                       params.with_neg_edges),
                    params)
        else:
            raise KeyError("Parameter not found")
    
    @staticmethod
    def get_model_data_by_params(params:BasicSettings):
        model_name = params.model_name
        data_name = params.data_name
        data = DataCFT(data_name)

        model_params = ModelDataParameterFactor.get_model_params(data, params)

        model =  CFT(*model_params)
        return model, data


    
