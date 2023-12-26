
import paddle
import torch

def torch2paddle():
    ckpt = torch.load("/home1/zhaoyh/paddlemodel/glow-tts_paddle/checkpoints/pretrained.pth")
    paddle_path = "/home1/zhaoyh/paddlemodel/glow-tts_paddle/checkpoints/pretrained.pdparams"
    paddle_state_dict = {}

    for param_name, param_value in ckpt["model"].items():
        v = param_value.cpu().numpy()
        if "weight_g" in param_name:
            print(param_name, v.shape)
            v = v.reshape(-1)
        
        param_name = param_name.replace("running_var", "_variance")
        param_name = param_name.replace("running_mean", "_mean")
        paddle_state_dict[param_name] = paddle.to_tensor(v)
    paddle_dict = {}
    paddle_dict['model'] = paddle_state_dict
    paddle_dict['iteration'] = ckpt['iteration']
    paddle_dict['learning_rate'] = ckpt['learning_rate']
    paddle.save(paddle_dict, paddle_path)

def hfiganconvert():
    ckpt = torch.load("/home1/zhaoyh/paddlemodel/glow-tts_paddle/checkpoints/LJ_V1/generator_v1")
    paddle_path = "/home1/zhaoyh/paddlemodel/glow-tts_paddle/checkpoints/LJ_V1/generator_v1.pdparams"
    paddle_state_dict = {}

    for param_name, param_value in ckpt["generator"].items():
        v = param_value.cpu().numpy()
        if "weight_g" in param_name:
            print(param_name, v.shape)
            v = v.reshape(-1)
        
        param_name = param_name.replace("running_var", "_variance")
        param_name = param_name.replace("running_mean", "_mean")
        paddle_state_dict[param_name] = paddle.to_tensor(v)
    paddle_dict = {}
    paddle_dict['generator'] = paddle_state_dict
    paddle.save(paddle_dict, paddle_path)
# def loadpaddle():
#     print("*** load paddle ***")
#     paddle_path = "/home1/zhaoyh/paddlemodel/vits_paddle/checkpoints/pretrained_ljs.pdparams"
#     ckpt = paddle.load(paddle_path)
#     print(ckpt.keys())
hfiganconvert()
# loadpaddle()