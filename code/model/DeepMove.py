import torch

import torch.nn as nn
import torch.nn.functional as F


class DeepMove(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, args, device, use_cuda):
        super(DeepMove, self).__init__()
        # 默认参数
        # self.loc_emb_size = 256
        # self.tim_emb_size = 64
        # self.uid_emb_size = 64
        self.hidden_size = args.hidden_size
        self.rnn_type = args.rnn_type
        self.tim_emb_size = args.tim_emb_size
        self.loc_emb_size = args.loc_emb_size
        self.uid_emb_size = args.uid_emb_size
        self.use_cuda = use_cuda
        self.device = device

        # Multi-modal Embbedding Layer
        self.emb_loc = nn.Embedding(args.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(args.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(args.uid_size, self.uid_emb_size)

        # Attention Selector
        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.hidden_size)

        if self.rnn_type == "GRU":
            # Recurrent Layer
            self.rnn_cur = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
            # Candidate Generator
            self.rnn_his = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == "LSTM":
            # Recurrent Layer
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)
            # Candidate Generator
            self.rnn_his = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == "RNN":
            # Recurrent Layer
            self.rnn = nn.RNN(input_size, self.hidden_size, 1, batch_first=True)
            # Candidate Generator
            self.rnn_his = nn.RNN(input_size, self.hidden_size, 1, batch_first=True)

        # Fully Connected Layer
        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, args.loc_size)
        self.dropout = nn.Dropout(p=args.dropout_p)

    def forward(self, history_loc, history_tim, loc, tim, uid):
        h1_cur = torch.zeros(1, tim.size()[0], self.hidden_size, requires_grad=True)
        c1_cur = torch.zeros(1, tim.size()[0], self.hidden_size, requires_grad=True)
        h1_his = torch.zeros(1, history_tim.size()[0], self.hidden_size, requires_grad=True)
        c1_his = torch.zeros(1, history_tim.size()[0], self.hidden_size, requires_grad=True)
        if self.use_cuda:
            h1_cur = h1_cur.to(self.device)
            c1_cur = c1_cur.to(self.device)
            h1_his = h1_his.to(self.device)
            c1_his = c1_his.to(self.device)

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x_cur = torch.cat((loc_emb, tim_emb), 2)
        x_cur = self.dropout(x_cur)

        loc_emb_history = self.emb_loc(history_loc)
        tim_emb_history = self.emb_tim(history_tim)

        x_his = torch.cat((loc_emb_history, tim_emb_history), 2)
        x_his = self.dropout(x_his)

    
        if self.rnn_type == "GRU" or self.rnn_type == "RNN":
            out_state_cur, h1_cur = self.rnn_cur(x_cur, h1_cur)
            out_state_his, h1_his = self.rnn_cur(x_his, h1_his)
        elif self.rnn_type == "LSTM":
            out_state_cur, (h1_cur, c1_cur) = self.rnn_cur(x_cur, (h1_cur, c1_cur))
            out_state_his, (h1_his, c1_his) = self.rnn_cur(x_his, (h1_his, c1_his))

        out_state_cur = F.selu(out_state_cur)
        out_state_his = F.selu(out_state_his)
        
        context = self.attn(out_state_cur, out_state_his)
        
        out = torch.cat((out_state_cur, context), dim=2)

        uid_emb = self.emb_uid(uid).unsqueeze(1)
        uid_emb = uid_emb.repeat(1, loc.size()[1], 1)
        combined_out = torch.cat((out, uid_emb), dim=2)
        combined_out = self.dropout(combined_out)

        y = self.fc_final(combined_out)
        score = F.log_softmax(y, dim=2)
        return score


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, out_state, history):
        # out_state: [batch_size, state_len_cur, hidden_size]  这个output 是decoder的每个时间步输出的隐藏状态
        # history: [batch_size, state_len_his, hidden_size]
        # score就是求得当前时间步的输出output和所有输入相似性关系的一个得分score , 下面就是通过softmax把这个得分转成权重
        # 可学习矩阵W_s为全1矩阵
        out_state = torch.tanh(out_state)
        scores = torch.matmul(out_state, history.transpose(-1, -2))
        # 此时第二维度的数字全都变成了0-1之间的数， 越大表示当前的输出output与哪个相关程度越大
        scores = F.softmax(scores, dim=2)
        # [batch_size, state_len, hidden_size]
        attn = torch.matmul(scores, history)   
        return attn