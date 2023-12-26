import sys
import paddle
from model_vc import Generator
import time
import datetime

paddle.set_device('gpu')

class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""
        self.vcc_loader = vcc_loader
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.use_cuda = paddle.device.cuda.device_count() >= 1
        self.device = str('cuda:0' if self.use_cuda else 'cpu').replace('cuda',
            'gpu')
        self.log_step = config.log_step
        self.build_model()

    def build_model(self):
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq
            )
        self.g_optimizer = paddle.optimizer.Adam(parameters=self.G.
            parameters(), learning_rate=0.0001, weight_decay=0.0)

    def reset_grad(self):
        """Reset the gradient buffers."""
        """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        self.g_optimizer.clear_grad()

    def train(self):
        data_loader = self.vcc_loader
        keys = ['G/loss_id', 'G/loss_id_psnt', 'G/loss_cd']
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            # x_real = x_real.to(self.device)
            # emb_org = emb_org.to(self.device)
            self.G.train()
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org,
                emb_org)
            g_loss_id = paddle.nn.functional.mse_loss(input=x_real, label=
                x_identic)
            g_loss_id_psnt = paddle.nn.functional.mse_loss(input=x_real,
                label=x_identic_psnt)
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = paddle.nn.functional.l1_loss(input=code_real, label
                =code_reconst)
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = 'Elapsed [{}], Iteration [{}/{}]'.format(et, i + 1,
                    self.num_iters)
                for tag in keys:
                    log += ', {}: {:.4f}'.format(tag, loss[tag])
                print(log)
