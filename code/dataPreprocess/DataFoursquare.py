import pickle

import numpy as np

from datetime import datetime, timedelta, timezone

class DataFoursquare():
    def __init__(self, save_path="../data/processedData/", data_path="../data/rawData/FourSquare/dataset_TSMC2014_TKY.csv", trace_split="interval", 
                 freq="h", hour_gap_max=24, trace_min=5, trace_max=30, sessions_min=3, train_split=0.8, auto_save=True):
        self.DATA_PATH = data_path
        self.SAVE_PATH = save_path
        
        self.save_name = "foursquare_" + trace_split + "_" + freq + ".pk"
        
        self.trace_split = trace_split
        self.hour_gap_max = hour_gap_max
        self.sessions_min = sessions_min
        self.trace_min = trace_min
        self.trace_max = trace_max
        self.train_split = train_split
        self.freq = freq
        
        self.start_time = None
        
        self.uid_list = {}
        # self.vid_list = {"pad": [0, 0, 0]}
        self.vid_list = {}
        self.data_neural = {}

        if auto_save == True:
            data = self.load_trajectory()
            data_sessions = self.split_trajectory(data)
            self.build_users_dict(data_sessions)
            self.build_locations_dict(data_sessions)
            self.prepare_neural_data(data_sessions)
            print("The number of user: " + str(len(self.uid_list)))
            print("The number of locations: " + str(len(self.vid_list)))
            print("The number of records: " + str(self.records_num(data_sessions)))
            self.save_variables()


    def load_trajectory(self): 
        count = 0
        data = {}        
        with open(self.DATA_PATH, "r") as f:
            # 数据集格式如下：
            # "User ID", "Venue ID", "Venue category ID", "Venue category name", 
            # "Latitude", "Longitude", "Timezone offset in minutes", "UTC time"
            attrs = f.readline()
            for line in f.readlines():
                attrs = line.strip("\n").split(",")
                # 用户id
                uid = int(attrs[0])
                # 地点经度
                lat = float(attrs[4])
                # 地点维度
                lng = float(attrs[5])
                # 时间的偏移
                timezone_offset = int(attrs[6])

                # 转换UTC时间戳为当地时区时间
                utc_datetime = datetime.strptime(attrs[7],"%a %b %d %H:%M:%S %z %Y")
                # timezone() 可以设置时区对象；timedelta() 是用来实现对日期的加减；在 UTC 的时间基础上，用 astimezone() 修改时区
                local_datetime = utc_datetime.astimezone(timezone(timedelta(minutes=timezone_offset)))

                # 获取数据集中的最开始的时间
                if(self.start_time is None or ((local_datetime - self.start_time).total_seconds() < 0)):
                    self.start_time = local_datetime
                
                count += 1
                if uid not in data:
                    data[uid] = []
                data[uid].append((lat, lng, local_datetime))


                # 此时data = {
                #     user : {
                #         [(lat, lng, local_datetime), (lat, lng, local_datetime), ……]
                #     }
                #     ...
                # }

        print("The number of original records: " + str(count))
        return data


    def split_trajectory(self, data):
        print("Split trajectory ...")
        if self.trace_split == "day":
            return self.split_trajectory_by_day(data)
        elif self.trace_split == "interval":
            return self.split_trajectory_by_interval(data)


    def split_trajectory_by_day(self, data):
        data_sessions = {}
        for u in data: 
            # 按天划分session
            traces_by_day = {}
            # enumerate 返回的是一个enumerate对象，用它可以同时获得索引(index)和值(value)

            for i, r in enumerate(data[u]):
                # %y 两位数的年份表示（00-99）
                # %m 月份（01-12）
                # %d 月内中的一天（0-31）
                # %a 本地简化星期名称
                time_str = data[u][i][2].strftime("%y%m%d-%a")
                if time_str not in traces_by_day:
                    traces_by_day[time_str] = []
                traces_by_day[time_str].append(r)
            
            # 划分为session_id
            days = list(traces_by_day.keys())
            days.sort()
            sessions = {}
            for tid in days:
                # 筛选一天中轨迹点数大于trace_min的天数
                if len(traces_by_day[tid]) >= self.trace_min:
                    sid = len(sessions)
                    sessions[sid] = traces_by_day[tid]

            # 筛选sessions数大于sessions_min的user
            if len(sessions) >= self.sessions_min: 
                data_sessions[u] = sessions

            # 此时data_sessions = {
            #     user_0 : {
            #         day_0 : [(lat, lng, local_datetime, category), (lat, lng, local_datetime, category), ……],
            #         day_1 : [(lat, lng, local_datetime, category), (lat, lng, local_datetime, category), ……],
            #     }
            #     ...
            # }
        return data_sessions


    def split_trajectory_by_interval(self, data):
        data_sessions = {}
        for u in data:
            # 子轨迹集，把每个用户的长轨迹分为若干个短的子轨迹，数据格式如下
            # session = {0 : [(lat, lng, time), (lat, lng, time), ……],
            #           1 : [(lat, lng, time), (lat, lng, time), ……],
            #           ……
            # }

            sessions = {}
            for i, r in enumerate(data[u]):
                time = r[2]
                sid = len(sessions)
                if i == 0:
                    sessions[sid] = [r]
                    last_time = time
                else:
                    interval = (time - last_time).total_seconds() // 3600
                    # 同一session中连续记录的时间间隔不大于hour_gap_max，且session长度不大于trace_max
                    if interval > self.hour_gap_max or len(sessions[sid - 1]) > self.trace_max:
                        sessions[sid] = [r]
                    else:
                        sessions[sid - 1].append(r)
                    last_time = time                

            sessions_filter = {}
            for s in sessions:
                # 筛选长度大于trace_min的session
                if len(sessions[s]) >= self.trace_min:
                    sessions_filter[len(sessions_filter)] = sessions[s]

            # 筛选sessions数大于sessions_min的user
            if len(sessions_filter) >= self.sessions_min: 
                data_sessions[u] = sessions_filter


            # 此时data_sessions = {
            #     user_0 : {
            #         0 : [(lat, lng, local_datetime), (lat, lng, local_datetime), ……],
            #         1 : [(lat, lng, local_datetime), (lat, lng, local_datetime), ……],
            #         ……
            #     }
            #     user_1 : {
            #         0 : [(lat, lng, local_datetime), (lat, lng, local_datetime), ……],
            #         1 : [(lat, lng, local_datetime), (lat, lng, local_datetime), ……],
            #         ……
            #     }
            #     ...
            # }
            
        return data_sessions

    def records_num(self, data_sessions):
        count = 0
        for u in data_sessions:
            for sid in data_sessions[u]:
                count += len(data_sessions[u][sid])
        return count


    def build_users_dict(self, data_sessions):
        print("Build Users dict ...")
        for u in data_sessions:
            if u not in self.uid_list:
                # uid_list： "user": [uid, 用户session数]
                self.uid_list[u] = [len(self.uid_list), len(data_sessions[u])]


    def build_locations_dict(self, data_sessions):
        print("Build locations dict ...")
        for u in data_sessions:
            sessions = data_sessions[u]
            sessions_id = sessions.keys()
            # train_id, test_id = split_train_test(list(sessions.keys()), self.train_split)

            for sid in sessions_id:
                session = sessions[sid]             
                for r in session:
                    lat_lon = (r[0], r[1])

                    if lat_lon not in self.vid_list:
                        # "[longitude, latitude]": [vid, number of location be visited]
                        self.vid_list[lat_lon] = [len(self.vid_list), 1]
                    else:
                        self.vid_list[lat_lon][1] += 1

        print(len(self.vid_list))

    # 时间编码
    # def trans_time(self, tm):
    #     minutes = (tm - self.start_time).total_seconds() // 3600 / 1.0
    #     tf = time_features(tm, freq=self.freq)
    #     tf.append(minutes)
    #     return tf
    
    # 转化datetime变量为整数，以周为周期，粒度为小时
    def tid_list(self, tm):
        # tid = tm.weekday() * 24 * 60 + tm.hour * 60 + tm.minute
        
        # tm = datetime.strftime(tm, "%a %b %d %H:%M:%S %z %Y")
        tid = tm.weekday() * 24 + tm.hour
        return tid


    def prepare_neural_data(self, data_sessions):
        # data_sessions = {
        #     user_0 : {
        #         0 : [(lat, lng, local_datetime), (lat, lng, local_datetime), ……],
        #         1 : [(lat, lng, local_datetime), (lat, lng, local_datetime), ……],
        #         ……
        #     }
        #     user_1 : {
        #         0 : [(lat, lng, local_datetime), (lat, lng, local_datetime), ……],
        #         1 : [(lat, lng, local_datetime), (lat, lng, local_datetime), ……],
        #         ……
        #     }
        #     ...
        # }
        for u in data_sessions:
            sessions = data_sessions[u]
            session_id = list(sessions.keys())
            split_id = int(np.floor(self.train_split * len(session_id)))
            train_id = session_id[:split_id]
            test_id = session_id[split_id:]
            sessions_tran = {}

            for sid in session_id:
                session = sessions[sid] 
                session_tran = []
                for r in session:
                    lat_lon = (r[0], r[1])
                    tm = r[2]
                    # 对时间和位置进行编码
                    vid = self.vid_list[lat_lon][0]
                    tid = self.tid_list(tm)

                    session_tran.append([vid, tid])
                sessions_tran[sid] = session_tran
            
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
            self.data_neural[self.uid_list[u][0]] = {"sessions": sessions_tran, "train": train_id, "test": test_id}


    # save variables
    def get_parameters(self):
        parameters = {}
        parameters["DATA_PATH"] = self.DATA_PATH
        parameters["SAVE_PATH"] = self.SAVE_PATH

        parameters["hour_gap_max"] = self.hour_gap_max
        parameters["trace_max"] = self.trace_max
        parameters["trace_min"] = self.trace_min
        parameters["sessions_min"] = self.sessions_min
        parameters["train_split"] = self.train_split
        return parameters

    def save_variables(self):
        foursquare_dataset = {"data_neural": self.data_neural, "vid_list": self.vid_list, 
                              "uid_list": self.uid_list, "parameters": self.get_parameters()}
        pickle.dump(foursquare_dataset, open(self.SAVE_PATH + self.save_name, "wb"))

# 注：
# python中时间日期格式化符号
# %y 两位数的年份表示（00-99）
# %Y 四位数的年份表示（000-9999）
# %m 月份（01-12）
# %d 月内中的一天（0-31）
# %H 24小时制小时数（0-23）
# %I 12小时制小时数（01-12）
# %M 分钟数（00=59）
# %S 秒（00-59）
# %a 本地简化星期名称
# %A 本地完整星期名称
# %b 本地简化的月份名称
# %B 本地完整的月份名称
# %c 本地相应的日期表示和时间表示
# %j 年内的一天（001-366）
# %p 本地A.M.或P.M.的等价符
# %U 一年中的星期数（00-53）星期天为星期的开始
# %w 星期（0-6），星期天为星期的开始
# %W 一年中的星期数（00-53）星期一为星期的开始
# %x 本地相应的日期表示
# %X 本地相应的时间表示
# %Z 当前时区的名称