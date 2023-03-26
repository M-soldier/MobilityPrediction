import torch
import pickle

import numpy as np

from torch.utils.data import Dataset
from dataPreprocess.DataFoursquare import DataFoursquare


class Dataset_Trajectory(Dataset):
    def __init__(self, root_path, model, trace_split="interval", flag="train", freq="h"):
        assert model in ["Markov", "RNN", "DeepMove"]
        # assert flag in ["train", "test"]

        self.root_path = root_path
        self.trace_split = trace_split
        self.mode = model
        self.flag = flag
        self.freq = freq
        
        self.dataset = []
        self.__read_data__()

    def __read_data__(self):     
        data_path = self.root_path + "foursquare_" + self.trace_split + "_" + self.freq + ".pk"
        # print(data_path)
        try:
            data = pickle.load(open(data_path, "rb+"))
            data_neural = data["data_neural"]
        except:
            loader = DataFoursquare(save_path=self.root_path, trace_split=self.trace_split, freq=self.freq)
            data_neural = loader.data_neural

        # data_neural = {
        #       user_id_0 : {
        #            sessions : {
        #               session_id_0 : [[vid, tid], ……],
        #               session_id_1 : [[vid, tid], ……],
        #               ……
        #           },
        #           train : [0, 1, 2, ……, k-1],
        #           test : [k, k+1, k+2, ……],
        #       }
        #       user_id_1 ……
        # }

        if self.mode == "Markov":
            self.dataset.append(data_neural)
        elif self.mode == "RNN":
            self.data_rnn(data_neural)
        elif self.mode == "DeepMove":
            self.data_deepmove(data_neural)


    # def data_markov(self, data_neural):
    #     for u in data_neural.keys():
    #         traces = data_neural[u]["sessions"]
    #         train_id = data_neural[u]["train"]
    #         test_id = data_neural[u]["test"]
            
    #         # 训练集的位置序列
    #         trace_train = []
    #         for tr in train_id:
    #             trace_train.append([t[0] for t in traces[tr]])
    #         locations_train = []
    #         for t in trace_train:
    #             locations_train.extend(t)

    #         # 测试集的位置序列
    #         trace_test = []
    #         for tr in test_id:
    #             trace_test.append([t[0] for t in traces[tr]])
    #         locations_test = []
    #         for t in trace_test:
    #             locations_test.extend(t)

    #         self.dataset.append([locations_train, locations_test])

    def data_rnn(self, data_neural):
        users = data_neural.keys()
        for u in users:
            sessions = data_neural[u]["sessions"]
            session_id = data_neural[u][self.flag]
            if self.flag == "test":
                for i in session_id:
                    session = sessions[i]
                    # 测试的时候按照子轨迹来测试
                    tim_np = np.array([s[1] for s in session[:-1]])
                    loc_np = np.array([s[0] + 1 for s in session[:-1]])
                    # 因为要预测下一个位置，所以要错开一位
                    target = np.array([s[0] + 1 for s in session[1:]])
                    trace = {}
                    trace["loc"] = torch.LongTensor(loc_np)
                    trace["tim"] = torch.LongTensor(tim_np)
                    trace["target"] = torch.LongTensor(target)
                    self.dataset.append(trace)
            elif self.flag == "train":
                tim_np = []
                loc_np = []
                target = []
                for i in session_id:
                    session = sessions[i]
                    # 将每个用户的子轨迹序列重新拼接成一个长序列训练
                    # print(session[0][1])
                    tim_np.extend([s[1] + 1 for s in session[:-1]])
                    loc_np.extend([s[0] + 1 for s in session[:-1]])
                    target.extend([s[0] + 1 for s in session[1:]])
                tim_np = np.array(tim_np)
                loc_np = np.array(loc_np)
                target = np.array(target)
                trace = {}
                trace["loc"] = torch.LongTensor(loc_np)
                trace["tim"] = torch.LongTensor(tim_np)
                trace["target"] = torch.LongTensor(target)
                self.dataset.append(trace)
        
    
    def data_deepmove(self, data_neural):
        for u in data_neural:
            sessions = data_neural[u]["sessions"]
            session_id = data_neural[u][self.flag]
            for i in session_id:
                if i == 0:
                    continue
                session = sessions[i]
                history_loc, history_tim = self.get_history(data_neural[u], i)

                tim_np = np.array([s[1] for s in session[:-1]])
                loc_np = np.array([s[0] for s in session[:-1]])
                # 因为要预测下一个位置，所以要错开一位
                target = np.array([s[0] for s in session[1:]])

                trace = {}
                trace["loc"] = torch.LongTensor(loc_np)
                trace["tim"] = torch.LongTensor(tim_np)
                trace["target"] = torch.LongTensor(target)
                trace["uid"] = torch.tensor(u)
                trace["history_loc"] = torch.LongTensor(history_loc)
                trace["history_tim"] = torch.LongTensor(history_tim)              

                self.dataset.append(trace)


    def get_history(self, user_neural, session_id):
        # user_neural = {
        #     sessions : {
        #         session_id_0 : [[vid, tid], ……],
        #         session_id_1 : [[vid, tid], ……],
        #         ……
        #     },
        #     train : [0, 1, 2, ……, k-1],
        #     test : [k, k+1, k+2, ……],
        # }
        sessions = user_neural["sessions"]
        history = []
        for sid in sessions:
            if sid < session_id:
                history.extend([(s[0], s[1]) for s in sessions[sid]])   
        history_loc = np.array([s[0] for s in history])
        history_tim = np.array([s[1] for s in history])

        return history_loc, history_tim


    def __getitem__(self, index):
        if self.mode == "RNN":
            return {
                "loc": self.dataset[index]["loc"],
                "tim": self.dataset[index]["tim"],
                "target": self.dataset[index]["target"]
            }
        elif self.mode == "DeepMove":
            return {
                "loc": self.dataset[index]["loc"],
                "tim": self.dataset[index]["tim"],
                "target": self.dataset[index]["target"],
                "uid": self.dataset[index]["uid"],
                "history_loc": self.dataset[index]["history_loc"],
                "history_tim": self.dataset[index]["history_tim"]
            }

    def __len__(self):
        return len(self.dataset)