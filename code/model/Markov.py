import numpy as np

def create_validation(dataset):
    validation = {}
    for u in dataset.keys():
        traces = dataset[u]["sessions"]
        train_id = dataset[u]["train"]
        test_id = dataset[u]["test"]
        # 训练集的位置序列
        trace_train = []
        for tr in train_id:
            trace_train.append([t[0] for t in traces[tr]])
        locations_train = []
        for t in trace_train:
            locations_train.extend(t)
        # 测试集的位置序列
        trace_test = []
        for tr in test_id:
            trace_test.append([t[0] for t in traces[tr]])
        locations_test = []
        for t in trace_test:
            locations_test.extend(t)
        validation[u] = [locations_train, locations_test]
    return validation

def Markov_1(data_neural):
    validation = create_validation(data_neural)        
    acc = 0
    count = 0
    user_acc = {}
    for u in validation.keys():
        # 训练集所有位置
        topk = list(set(validation[u][0])) 
        # 转移矩阵
        transfer = np.zeros((len(topk), len(topk)))

        # train
        sessions = data_neural[u]["sessions"]
        train_id = data_neural[u]["train"]
        for i in train_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]  # 当前位置j
                target = sessions[i][j + 1][0]  # 下一个位置
                if loc in topk and target in topk:
                    r = topk.index(loc)     # 对应的索引
                    c = topk.index(target)  
                    transfer[r, c] += 1
        for i in range(len(topk)):
            tmp_sum = np.sum(transfer[i, :])
            if tmp_sum > 0:
                transfer[i, :] = transfer[i, :] / tmp_sum   

        # validation
        user_count = 0
        user_acc[u] = 0
        test_id = data_neural[u]["test"]
        for i in test_id:
            for j, s in enumerate(sessions[i][:-1]):
                loc = s[0]
                target = sessions[i][j + 1][0]
                count += 1
                user_count += 1
                if loc in topk:
                    # 预测取条件概率最大的位置
                    pred = np.argmax(transfer[topk.index(loc), :])
                    if pred >= len(topk) - 1:
                        pred = np.random.randint(len(topk))
                    pred2 = topk[pred]
                    # 计算准确率
                    if pred2 == target:
                        acc += 1
                        user_acc[u] += 1
        user_acc[u] = user_acc[u] / user_count
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    print("First order markov accuracy of the model on the test dataset : {}%.".format(avg_acc*100))
    return avg_acc, user_acc

def Markov_2(data_neural):
    validation = create_validation(data_neural)

    acc = 0
    count = 0
    user_acc = {}
    for u in validation.keys():
        # 训练集所有位置
        topk = list(set(validation[u][0]))
        # 转移矩阵
        transfer = np.zeros((len(topk), len(topk), len(topk)))

        # train
        sessions = data_neural[u]["sessions"]
        train_id = data_neural[u]["train"]
        for i in train_id:
            for j, s in enumerate(sessions[i][:-2]):
                loc_1 = s[0]  # 当前位置j
                loc_2 = sessions[i][j + 1][0]
                target = sessions[i][j + 2][0]  # 下一个位置
                if loc_1 in topk and loc_2 in topk and target in topk:
                    m = topk.index(loc_1)     # 对应的索引
                    n = topk.index(loc_2)
                    c = topk.index(target)  
                    transfer[m, n, c] += 1
        for i in range(len(topk)):
            for j in range(len(topk)):
                tmp_sum = np.sum(transfer[i, j, :])
            if tmp_sum > 0:
                transfer[i, j, :] = transfer[i, j, :] / tmp_sum   

        # validation
        user_count = 0
        user_acc[u] = 0
        test_id = data_neural[u]["test"]
        for i in test_id:
            for j, s in enumerate(sessions[i][:-2]):
                loc_1 = s[0]
                loc_2 = sessions[i][j + 1][0]
                target = sessions[i][j + 2][0]
                count += 1
                user_count += 1
                if loc_1 in topk and loc_2 in topk:
                    # 预测取条件概率最大的位置
                    pred = np.argmax(transfer[topk.index(loc_1), topk.index(loc_2), :])
                    if pred >= len(topk) - 1:
                        pred = np.random.randint(len(topk))
                    pred2 = topk[pred]
                    # 计算准确率
                    if pred2 == target:
                        acc += 1
                        user_acc[u] += 1
        user_acc[u] = user_acc[u] / user_count
    avg_acc = np.mean([user_acc[u] for u in user_acc])
    print("Second order markov Accuracy of the model on the test dataset : {}%.".format(avg_acc*100))
    return avg_acc, user_acc