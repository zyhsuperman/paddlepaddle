import numpy as np
import torch
import paddle

# landmark generator
# torch_path = "/home/zhaoyh/talkingfacemodel/PaddleIP_LAP/checkpoints/Landmark Generator Checkpoint.pth"
# paddle_path = "/home/zhaoyh/talkingfacemodel/PaddleIP_LAP/checkpoints/Landmark Generator Checkpoint.pdparams"
# ckpt = torch.load(torch_path)
# paddle_state_dict = {}
# for name, param in ckpt['state_dict'].items():
#     if "num_batches_tracked" in name:
#         continue
#     name = name.replace("running_var", "_variance")
#     name = name.replace("running_mean", "_mean")
#     if "self_attn.in_proj_weight" in name:
#         v_q_w, v_k_w, v_v_w = param.chunk(3, dim=0)
#         v_q_w = v_q_w.detach().cpu().transpose(0, 1).numpy()
#         v_k_w = v_k_w.detach().cpu().transpose(0, 1).numpy()
#         v_v_w = v_v_w.detach().cpu().transpose(0, 1).numpy()
#         paddle_state_dict[name.replace("self_attn.in_proj_weight", "self_attn.q_proj.weight")] = v_q_w
#         paddle_state_dict[name.replace("self_attn.in_proj_weight", "self_attn.k_proj.weight")] = v_k_w
#         paddle_state_dict[name.replace("self_attn.in_proj_weight", "self_attn.v_proj.weight")] = v_v_w
#         print(name.replace("self_attn.in_proj_weight", "self_attn.q_proj.weight"))
#         print(name.replace("self_attn.in_proj_weight", "self_attn.k_proj.weight"))
#         print(name.replace("self_attn.in_proj_weight", "self_attn.v_proj.weight"))
#     elif "self_attn.in_proj_bias" in name:
#         v_q_b, v_k_b, v_v_b = param.chunk(3, dim=0)
#         v_q_b = v_q_b.detach().cpu().numpy()
#         v_k_b = v_k_b.detach().cpu().numpy()
#         v_v_b = v_v_b.detach().cpu().numpy()
#         paddle_state_dict[name.replace("self_attn.in_proj_bias", "self_attn.q_proj.bias")] = v_q_b
#         paddle_state_dict[name.replace("self_attn.in_proj_bias", "self_attn.k_proj.bias")] = v_k_b
#         paddle_state_dict[name.replace("self_attn.in_proj_bias", "self_attn.v_proj.bias")] = v_v_b
#         print(name.replace("self_attn.in_proj_bias", "self_attn.q_proj.bias"))
#         print(name.replace("self_attn.in_proj_bias", "self_attn.k_proj.bias"))
#         print(name.replace("self_attn.in_proj_bias", "self_attn.v_proj.bias"))
#     elif "self_attn.out_proj.weight"  in name:
#         print(name)
#         v = param.detach().cpu().transpose(0, 1).numpy()
#     elif "linear1.weight" in name:
#         print(name)
#         v = param.detach().cpu().transpose(0, 1).numpy()
#     elif "linear2.weight" in name:
#         print(name)
#         v = param.detach().cpu().transpose(0, 1).numpy()
#     elif "mouse_keypoint_map.weight" in name:
#         print(name)
#         v = param.detach().cpu().transpose(0, 1).numpy()
#     elif "jaw_keypoint_map.weight" in name:
#         print(name)
#         v = param.detach().cpu().transpose(0, 1).numpy()
#     else:
#         v = param.detach().cpu().numpy()
#     paddle_state_dict[name] = v
# paddle_dict = {}
# paddle_dict['state_dict'] = paddle_state_dict
# paddle.save(paddle_dict, paddle_path)

# # render
torch_path = "/home/zhaoyh/talkingfacemodel/PaddleIP_LAP/checkpoints/CVPR2023pretrain models renderer.pth"
paddle_path = "/home/zhaoyh/talkingfacemodel/PaddleIP_LAP/checkpoints/CVPR2023pretrain models renderer.pdparams"
ckpt = torch.load(torch_path)
paddle_state_dict = {}
for name, param in ckpt['state_dict'].items():
    if "num_batches_tracked" in name:
        continue
    name = name.replace("running_var", "_variance")
    name = name.replace("running_mean", "_mean")
    name = name.replace("module.", "", 1)
    if "mlp_gamma.weight" in name:
        print(name)
        v = param.detach().cpu().transpose(0, 1).numpy()
    elif "mlp_shared.0.weight" in name:
        print(name)
        v = param.detach().cpu().transpose(0, 1).numpy()
    elif "mlp_beta.weight" in name:
        print(name)
        v = param.detach().cpu().transpose(0, 1).numpy()
    else:
        v = param.detach().cpu().numpy()
    paddle_state_dict[name] = v
paddle_dict = {}
paddle_dict['state_dict'] = paddle_state_dict
paddle.save(paddle_dict, paddle_path)