import torch

import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    """rnn model with long-term history attention"""

    # def __init__(self, loc_size, tim_size, batch_size, rnn_type, device, use_cuda, 
    #              tim_emb_size, loc_emb_size, hidden_size, dropout_p):
    def __init__(self, args, device, use_cuda):
        super(RNN, self).__init__()
        self.loc_size = args.loc_size
        self.tim_size = args.tim_size
        self.tim_emb_size = args.tim_emb_size
        self.loc_emb_size = args.loc_emb_size
        self.hidden_size = args.hidden_size
        self.rnn_type = args.rnn_type
        self.use_cuda = use_cuda
        self.device = device
        
        self.emb_loc = nn.Embedding(self.loc_size + 1, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size + 1, self.tim_emb_size)
        # self.emb_tim = TimeFeatureEmbedding(d_time=self.tim_emb_size, freq="h")

        input_size = self.loc_emb_size + self.tim_emb_size
        # print(self.loc_emb_size)
        # print(self.tim_emb_size)
        # print("input_size", input_size)

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(input_size, self.hidden_size, 1, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        # 在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态，模拟现实生活中的某些频道的数据缺失
        # 达到数据增强和减少过拟合的效果
        # 在测试阶段，通过model.eval()将其关闭
        self.dropout = nn.Dropout(p=args.dropout_p)

    def forward(self, loc, tim):
        # print(tim.size())
        h1 = torch.zeros(1, tim.size()[0], self.hidden_size, requires_grad=True)
        c1 = torch.zeros(1, tim.size()[0], self.hidden_size, requires_grad=True)
        if self.use_cuda:
            h1 = h1.to(self.device)
            c1 = c1.to(self.device)

        # embedding之后 tim_emb是[sequence length, batch size, embedding size]维度
        tim_emb = self.emb_tim(tim)
        loc_emb = self.emb_loc(loc)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)
        # print(x.size())
        if self.rnn_type == "GRU" or self.rnn_type == "RNN":
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == "LSTM":
            out, (h1, c1) = self.rnn(x, (h1, c1))

        out = F.selu(out)
        out = self.dropout(out)
        
        y = self.fc(out)
        score = F.log_softmax(y, dim=2)
        return score