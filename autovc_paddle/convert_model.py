
import paddle
import torch

def wavebetTorch2paddle():
    ckpt = torch.load("//home1/zhaoyh/paddlemodel/autovc_paddle/checkpoints/checkpoint_step001000000_ema.pth")
    paddle_path = "/home1/zhaoyh/paddlemodel/autovc_paddle/checkpoints/checkpoint_step001000000_ema.pdparams"
    paddle_state_dict = {}

    for param_name, param_value in ckpt["state_dict"].items():
        v = param_value.cpu().numpy()
        if "weight_g" in param_name:
            print(param_name, v.shape)
            v = v.reshape(-1)
        
        param_name = param_name.replace("running_var", "_variance")
        param_name = param_name.replace("running_mean", "_mean")
        paddle_state_dict[param_name] = paddle.to_tensor(v)
    paddle_dict = {}
    paddle_dict['state_dict'] = paddle_state_dict
    paddle_dict['global_step'] = ckpt['global_step']
    paddle_dict['global_epoch'] = ckpt['global_epoch']
    paddle_dict['global_test_step'] = ckpt['global_test_step']
    paddle.save(paddle_dict, paddle_path)

def loadpaddle(paddle_path):
    print("*** load paddle ***")
    ckpt = paddle.load(paddle_path)
    print(ckpt.keys())

def sencoderTorch2paddle():
    ckpt = torch.load("/home1/zhaoyh/paddlemodel/autovc_paddle/checkpoints/3000000-BL.ckpt")
    paddle_path = "/home1/zhaoyh/paddlemodel/autovc_paddle/checkpoints/3000000-BL_1.pdparams"
    paddle_state_dict = {}

    for param_name, param_value in ckpt["model_b"].items():
        v = param_value.cpu().numpy()
        # if "lstm.weight_ih_l0" in param_name:
        #     new_param_name = "module.lstm.0.cell.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.weight_hh_l0" in param_name:
        #     new_param_name = "module.lstm.0.cell.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.bias_ih_l0" in param_name:
        #     new_param_name = "module.lstm.0.cell.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.bias_hh_l0" in param_name:
        #     new_param_name = "module.lstm.0.cell.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.weight_ih_l1" in param_name:
        #     new_param_name = "module.lstm.1.cell.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.weight_hh_l1" in param_name:
        #     new_param_name = "module.lstm.1.cell.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.bias_ih_l1" in param_name:
        #     new_param_name = "module.lstm.1.cell.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.bias_hh_l1" in param_name:
        #     new_param_name = "module.lstm.1.cell.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.weight_ih_l2" in param_name:
        #     new_param_name = "module.lstm.2.cell.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.weight_hh_l2" in param_name:
        #     new_param_name = "module.lstm.2.cell.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.bias_ih_l2" in param_name:
        #     new_param_name = "module.lstm.2.cell.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "lstm.bias_hh_l2" in param_name:
        #     new_param_name = "module.lstm.2.cell.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)

        if "module.embedding.weight" in param_name:
            v = v.T
            print(param_name, v.shape)
        
        param_name = param_name.replace("running_var", "_variance")
        param_name = param_name.replace("running_mean", "_mean")
        paddle_state_dict[param_name] = paddle.to_tensor(v)
    paddle_dict = {}
    paddle_dict['model_b'] = paddle_state_dict
    paddle.save(paddle_dict, paddle_path)

def autovctorch2paddle():
    ckpt = torch.load("/home1/zhaoyh/paddlemodel/autovc_paddle/checkpoints/autovc.ckpt")
    paddle_path = "/home1/zhaoyh/paddlemodel/autovc_paddle/checkpoints/autovc_1.pdparams"
    paddle_state_dict = {}
    for param_name, param_value in ckpt["model"].items():
        v = param_value.cpu().numpy()
        if "num_batches_tracked" in param_name:  # 飞桨中无此参数，无需保存
            continue

        # if "encoder.lstm.weight_ih_l0" in param_name:
        #     new_param_name = "encoder.lstm.0.cell_fw.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.weight_hh_l0" in param_name:
        #     new_param_name = "encoder.lstm.0.cell_fw.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.bias_ih_l0" in param_name:
        #     new_param_name = "encoder.lstm.0.cell_fw.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.bias_hh_l0" in param_name:
        #     new_param_name = "encoder.lstm.0.cell_fw.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.weight_ih_l0_reverse" in param_name:
        #     new_param_name = "encoder.lstm.0.cell_bw.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.weight_hh_l0_reverse" in param_name:
        #     new_param_name = "encoder.lstm.0.cell_bw.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.bias_ih_l0_reverse" in param_name:
        #     new_param_name = "encoder.lstm.0.cell_bw.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.bias_hh_l0_reverse" in param_name:
        #     new_param_name = "encoder.lstm.0.cell_bw.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.weight_ih_l1" in param_name:
        #     new_param_name = "encoder.lstm.1.cell_fw.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.weight_hh_l1" in param_name:
        #     new_param_name = "encoder.lstm.1.cell_fw.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.bias_ih_l1" in param_name:
        #     new_param_name = "encoder.lstm.1.cell_fw.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.bias_hh_l1" in param_name:
        #     new_param_name = "encoder.lstm.1.cell_fw.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.weight_ih_l1_reverse" in param_name:
        #     new_param_name = "encoder.lstm.1.cell_bw.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.weight_hh_l1_reverse" in param_name:
        #     new_param_name = "encoder.lstm.1.cell_bw.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.bias_ih_l1_reverse" in param_name:
        #     new_param_name = "encoder.lstm.1.cell_bw.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "encoder.lstm.bias_hh_l1_reverse" in param_name:
        #     new_param_name = "encoder.lstm.1.cell_bw.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)

        # if "decoder.lstm1.weight_ih_l0" in param_name:
        #     new_param_name = "decoder.lstm1.0.cell.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v) 
        # if "decoder.lstm1.weight_hh_l0" in param_name:
        #     new_param_name = "decoder.lstm1.0.cell.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm1.bias_ih_l0" in param_name:
        #     new_param_name = "decoder.lstm1.0.cell.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm1.bias_hh_l0" in param_name:
        #     new_param_name = "decoder.lstm1.0.cell.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm2.weight_ih_l0" in param_name:
        #     new_param_name = "decoder.lstm2.0.cell.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm2.weight_hh_l0" in param_name:
        #     new_param_name = "decoder.lstm2.0.cell.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm2.bias_ih_l0" in param_name:
        #     new_param_name = "decoder.lstm2.0.cell.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm2.bias_hh_l0" in param_name:
        #     new_param_name = "decoder.lstm2.0.cell.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm2.weight_ih_l1" in param_name:
        #     new_param_name = "decoder.lstm2.1.cell.weight_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm2.weight_hh_l1" in param_name:
        #     new_param_name = "decoder.lstm2.1.cell.weight_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm2.bias_ih_l1" in param_name:
        #     new_param_name = "decoder.lstm2.1.cell.bias_ih"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)
        # if "decoder.lstm2.bias_hh_l1" in param_name:
        #     new_param_name = "decoder.lstm2.1.cell.bias_hh"
        #     paddle_state_dict[new_param_name] = paddle.to_tensor(v)



        if "decoder.linear_projection.linear_layer.weight" in param_name:
            v = v.T
            print(param_name, v.shape)
        
        param_name = param_name.replace("running_var", "_variance")
        param_name = param_name.replace("running_mean", "_mean")
        paddle_state_dict[param_name] = paddle.to_tensor(v)
    paddle_dict = {}
    paddle_dict['model'] = paddle_state_dict
    paddle.save(paddle_dict, paddle_path)
sencoderTorch2paddle()
autovctorch2paddle()
# torch2paddle()
# loadpaddle()