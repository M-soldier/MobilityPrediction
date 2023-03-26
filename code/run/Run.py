import os
import time
import json
import torch

import numpy as np
import torch.optim as Optim

from datetime import datetime

from model.RNN import RNN
from model.DeepMove import DeepMove
from model.Markov import Markov_1, Markov_2

from dataPreprocess.DataProvider import DataProvider


class Run():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.model == "RNN":
            if str(self.device) == "cuda":
                self.model = RNN(self.args, device=self.device, use_cuda=True).to(self.device)
            else:
                self.model = RNN(self.args, device=self.device, use_cuda=False).to(self.device)
        elif args.model == "DeepMove":
            if str(self.device) == "cuda":
                self.model = DeepMove(self.args, device=self.device, use_cuda=True).to(self.device)
            else:
                self.model = DeepMove(self.args, device=self.device, use_cuda=False).to(self.device)


    def _get_data(self, flag):
        if self.args.model == "Markov":
            data_set = DataProvider(self.args, None)
            return data_set
        else:
            data_set, data_loader = DataProvider(self.args, flag)
            return data_set, data_loader
    
    def get_acc(self, target, scores):
        target = target.data.cpu().numpy()
        # 求tensor中某个dim的前k大或者前k小的值以及对应的index
        _, idxx = scores.data.topk(10, 1)
        predx = idxx.cpu().numpy()
        acc = [0,0,0]
        count = 0
        for i, p in enumerate(predx):
            t = target[i]
            if t > 0:
                count += 1
            if t in p[:10] and t > 0:
                acc[0] += 1
            if t in p[:5] and t > 0:
                acc[1] += 1
            if t == p[0] and t > 0:
                acc[2] += 1
        return count,acc
        
    def run_simple(self, data_loader, mode, optimizer):
        if mode == "train":
            self.model.train()
        elif mode == "test":
            self.model.eval()
        total_loss = []
        batch_acc = {}
        # print("Total Batch :", data_loader.__len__())
        criterion = torch.nn.NLLLoss().to(self.device)

        for i,batch_list in enumerate(data_loader):
            # print("*" * 10, "Start Batch :",i , time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "*" * 10)
            optimizer.zero_grad()
            if self.args.model == "RNN":
                tim = batch_list["tim"].to(self.device)
                loc = batch_list["loc"].to(self.device)

                target = batch_list["target"].to(self.device)
                #loc和tim都是[sequence length, batch size]维度的tensor
                scores = self.model(loc, tim) 
            elif self.args.model == "DeepMove":
                tim = batch_list["tim"].to(self.device)
                loc = batch_list["loc"].to(self.device)
                uid = batch_list["uid"].to(self.device)
                history_loc = batch_list["history_loc"].to(self.device)
                history_tim = batch_list["history_tim"].to(self.device)
                target = batch_list["target"].to(self.device)
                scores = self.model(history_loc, history_tim, loc, tim, uid)


            if mode == "train":
                scores = scores.reshape(tim.size()[0]*tim.size()[1],-1)
                target = target.reshape(tim.size()[0]*tim.size()[1])

                # 使用nll_loss时，如果想计算batch的loss，假设loss函数输入x的shape为
                # (N, d, C)，其中N是batch_size，d是句子长度，C是vocab_size，标签target y的shape为(N, d)
                # nll_loss函数要求输入为 (N, C, d)，target为(N, d)，则计算时，需要将x的后两维做转置：
                # loss = torch.nn.functional.nll_loss(x.transpose(1, 2), y)         

                loss = criterion(scores, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optimizer.step()
            elif mode == "test":
                scores = scores.reshape(tim.size()[0]*tim.size()[1],-1)
                target = target.reshape(tim.size()[0]*tim.size()[1])

                loss = criterion(scores, target)
                batch_acc[i] = [0,0]
                batch_acc[i][0],acc = self.get_acc(target, scores)
                batch_acc[i][1] = acc[2]
                
            total_loss.append(loss.data.cpu().numpy())
            torch.cuda.empty_cache()
            
        avg_loss = np.mean(total_loss, dtype=np.float64)
        if mode == "train":
            return avg_loss
        elif mode == "test":
            tmp_0 = sum([batch_acc[s][0] for s in batch_acc])
            tmp_1 = sum([batch_acc[s][1] for s in batch_acc])
            avg_acc = tmp_1 / tmp_0
            return avg_loss, avg_acc
        

    def train(self):
        if self.args.model == "Markov":
            dataset = self._get_data(None)
            Markov_1(dataset)
            Markov_2(dataset)
        else:
            _, train_loader = self._get_data(flag="train")
            _, test_loader = self._get_data(flag="test")

            if self.args.optim == "Adam":
                optimizer = Optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate, weight_decay=self.args.L2)
            elif self.args.optim == "SGD":
                optimizer = Optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate, weight_decay=self.args.L2)
            
            scheduler = Optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=self.args.lr_step, factor=self.args.lr_decay, threshold=1e-4)
            metrics = {"train_loss": [], "valid_loss": [], "accuracy": []}
            save_path_tmp = self.args.save_path + "checkpoint/" + self.args.model + "/"
            save_path_res = self.args.save_path + "res/"
            save_path_model = self.args.save_path + "model/"

            if not os.path.exists(save_path_tmp):
                os.makedirs(save_path_tmp)
                
            if not os.path.exists(save_path_model):
                os.makedirs(save_path_model)
                
            if not os.path.exists(save_path_res):
                os.makedirs(save_path_res)

            for epoch in range(self.args.epoch_max):
                st = time.time()
                avg_loss = self.run_simple(train_loader, "train", optimizer)
                print("==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}".format(epoch, avg_loss, self.args.learning_rate))
                metrics["train_loss"].append(avg_loss)

                avg_loss, avg_acc = self.run_simple(test_loader, "test", optimizer)
                print("==>Test Acc:{:.4f} Loss:{:.4f}".format(avg_acc, avg_loss))

                metrics["valid_loss"].append(avg_loss)
                metrics["accuracy"].append(avg_acc)

                save_name_tmp = "ep_" + str(epoch) + ".m"
                torch.save(self.model.state_dict(), save_path_tmp + save_name_tmp)

                scheduler.step(avg_acc)
                lr_last = self.args.learning_rate
                self.args.learning_rate = optimizer.param_groups[0]["lr"]
                if lr_last > self.args.learning_rate:
                    load_epoch = np.argmax(metrics["accuracy"])
                    load_name_tmp = "ep_" + str(load_epoch) + ".m"
                    self.model.load_state_dict(torch.load(save_path_tmp + load_name_tmp))
                    print("load epoch={} model state".format(load_epoch))
                if epoch == 0:
                    print("*"*20)
                    print("single epoch time cost:{}".format(time.time() - st))
                if self.args.learning_rate <= 0.9 * 1e-6:
                    break

                print("*"*20)

            argv = {
                "model":self.args.model,
                "data":self.args.data,
                "root_path ":self.args.root_path ,
                "save_path":self.args.save_path,
                "loc_emb_size":self.args.loc_emb_size,
                "tim_emb_size":self.args.tim_emb_size,
                "hidden_size":self.args.hidden_size,
                "dropout_p":self.args.dropout_p,
                "data_name":self.args.data_name,
                "learning_rate":self.args.learning_rate,
                "batch_size":self.args.batch_size,
                "lr_step":self.args.lr_step,
                "lr_decay":self.args.lr_decay,
                "optim":self.args.optim,
                "L2":self.args.L2,
                "clip":self.args.clip,
                "epoch_max":self.args.epoch_max,
                "rnn_type ":self.args.rnn_type ,
                "trace_split":self.args.trace_split,
                "freq":self.args.freq,
                "loc_size":self.args.loc_size,
                "tim_size":self.args.tim_size,
                "uid_size":self.args.uid_size,
                "uid_emb_size":self.args.uid_emb_size,
            }
                
            mid = np.argmax(metrics["accuracy"])
            avg_acc = metrics["accuracy"][mid]
            load_name_tmp = "ep_" + str(mid) + ".m"
            self.model.load_state_dict(torch.load(save_path_tmp + load_name_tmp))
            save_name = self.args.model + "_" + self.args.rnn_type + "_" + datetime.now().strftime("%Y-%m-%d %H:%M")
            json.dump({"argv": argv, "metrics": metrics}, fp=open(save_path_res + "res_" + save_name + ".rs", "w"), indent=4)
            metrics_view = {"train_loss": [], "valid_loss": [], "accuracy": []}
            for key in metrics_view:
                metrics_view[key] = metrics[key]
            json.dump({"args": argv, "metrics": metrics_view}, fp=open(save_path_res + "res_" + save_name + ".txt", "w"), indent=4)
            torch.save(self.model.state_dict(), save_path_model + save_name + ".m")
