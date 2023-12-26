import pickle
import random
import numpy as np
# 替换为你的 .pkl 文件路径
file_path = '/home1/zhaoyh/paddlemodel/autovc_paddle/spmel/train.pkl'

# 以二进制读取模式打开文件
with open(file_path, 'rb') as file:
    # 使用 pickle 加载文件内容
    datas = pickle.load(file)

metadatas = []
for data in datas:
    metadata = []
    metadata.append(data[0])
    metadata.append(data[1])
    print(data[2])
    spec = np.load('/home1/zhaoyh/paddlemodel/autovc_paddle/spmel/' + random.choice(data[2:]))
    metadata.append(spec)
    metadatas.append(metadata)

with open('mymetadata.pkl', 'wb') as handle:
    pickle.dump(metadatas, handle)
