import torch

from torch.utils.data import DataLoader
from dataPreprocess.Dataset import Dataset_Trajectory

data_dict = {
    "Foursquare": Dataset_Trajectory,
}

def DataProvider(args, flag):
    Data = data_dict[args.data]
    # Data = Dataset_Trajectory(root_path=args.root_path, mode=args.mode, flag=flag)

    if flag == "test":
        shuffle_flag = False
        batch_size = args.batch_size
    elif flag == "train":
        shuffle_flag = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        model=args.model,
        trace_split=args.trace_split,
        freq=args.freq,
        flag=flag,
    )

    
    if args.model == "RNN":
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            collate_fn=collate_fn_rnn,
            # num_workers=args.num_workers
            num_workers=16
            )
        return data_set, data_loader
    elif args.model == "DeepMove":
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            collate_fn=collate_fn_deepmove,
            # num_workers=args.num_workers
            num_workers=16
            )
        return data_set, data_loader
    elif args.model == "Markov":
        return data_set.dataset[0]


# 定义rnn模型的dataloader每次迭代输出的数据形式
def collate_fn_rnn(batch):
    tim = [s["tim"] for s in batch]
    loc = [s["loc"] for s in batch]
    target = [s["target"] for s in batch]
    trandata = {}
    trandata["tim"] = torch.nn.utils.rnn.pad_sequence(tim, batch_first=True, padding_value=0)
    trandata["loc"] = torch.nn.utils.rnn.pad_sequence(loc, batch_first=True, padding_value=0)
    trandata["target"] = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)
    return trandata

# 定义deepmove模型的dataloader每次迭代输出的数据形式
def collate_fn_deepmove(batch):
    # return collate_batch(batch, pad_id=args.pad_id, max_padding=args.max_padding)
    loc = [s["loc"] for s in batch]
    tim = [s["tim"] for s in batch]
    target = [s["target"] for s in batch]
    uid = [s["uid"] for s in batch]
    history_loc = [s["history_loc"] for s in batch]
    history_tim = [s["history_tim"] for s in batch]

    trandata = {}
    
    trandata["tim"] = torch.nn.utils.rnn.pad_sequence(tim, batch_first=True, padding_value=0)
    trandata["loc"] = torch.nn.utils.rnn.pad_sequence(loc, batch_first=True, padding_value=0)
    trandata["target"] = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)
    trandata["history_loc"] = torch.nn.utils.rnn.pad_sequence(history_loc, batch_first=True, padding_value=0)
    trandata["history_tim"] = torch.nn.utils.rnn.pad_sequence(history_tim, batch_first=True, padding_value=0)
    trandata["uid"] = torch.stack(uid, dim=0)

    return trandata