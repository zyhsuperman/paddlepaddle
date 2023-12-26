import torch
import paddle
from models import Landmark_generator as Landmark_transformer
from models import Renderer

# #加载landmark_generator模型
# d_model = 512
# dim_feedforward = 1024
# nlayers = 4
# nhead = 4
# dropout = 0.1
# Nl = 15
# T = 5
# Project_name = 'landmarkT5_d512_fe1024_lay4_head4'
# model = Landmark_transformer(T, d_model, nlayers, nhead,
#     dim_feedforward, dropout)

# # 恢复模型参数
# checkpoint_path = '/home/zhaoyh/talkingfacemodel/PaddleIP_LAP/checkpoints/Landmark Generator Checkpoint.pdparams'
# model.set_state_dict(paddle.load(checkpoint_path)['state_dict'])

# for name, param in model.named_parameters():
#     print(name, param.shape)


# #输出权重文件
# checkpoints = torch.load('/home/zhaoyh/talkingfacemodel/PaddleIP_LAP/checkpoints/CVPR2023pretrain models renderer.pth')

# for param_name, tensor in checkpoints['state_dict'].items():
#     print(param_name, tensor.shape)

# #加载render模型
model=Renderer()
# 恢复模型参数
checkpoint_path = '/home/zhaoyh/talkingfacemodel/PaddleIP_LAP/checkpoints/CVPR2023pretrain models renderer.pdparams'
model.set_state_dict(paddle.load(checkpoint_path)['state_dict'])

# for name, param in model.named_parameters():
#     print(name, param.shape)

