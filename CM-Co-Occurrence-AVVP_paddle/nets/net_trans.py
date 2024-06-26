import paddle
import paddle.nn.functional as F
import numpy as np
import copy
import math
from ipdb import set_trace
from typing import Optional, Any
import os

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

paddle.seed(seed=0)
np.random.seed(0)
paddle.seed(seed=0)
paddle.seed(seed=0)


def _get_clones(module, N):
    return paddle.nn.LayerList(sublayers=[copy.deepcopy(module) for i in
        range(N)])


class Encoder(paddle.nn.Layer):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=512)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        output_a = src_a
        output_v = src_v
        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                src_key_padding_mask=src_key_padding_mask)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)
        return output_a, output_v


class HANLayer(paddle.nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = paddle.nn.MultiHeadAttention(d_model, nhead,
            dropout=dropout)
        self.cm_attn = paddle.nn.MultiHeadAttention(d_model, nhead, dropout=
            dropout)
        self.linear1 = paddle.nn.Linear(in_features=d_model, out_features=
            dim_feedforward)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.linear2 = paddle.nn.Linear(in_features=dim_feedforward,
            out_features=d_model)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.dropout11 = paddle.nn.Dropout(p=dropout)
        self.dropout12 = paddle.nn.Dropout(p=dropout)
        self.dropout2 = paddle.nn.Dropout(p=dropout)
        self.activation = paddle.nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

		Args:
			src: the sequnce to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
        src_q = src_q.transpose(perm=[1, 0, 2])
        src_v = src_v.transpose(perm=[1, 0, 2])
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask)[0]
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)
        src2 = self.linear2(self.dropout(paddle.nn.functional.relu(x=self.
            linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.transpose(perm=[1, 0, 2])


class MMIL_Net(paddle.nn.Layer):

    def __init__(self, opt):
        super(MMIL_Net, self).__init__()
        self.fc_prob = paddle.nn.Linear(in_features=512, out_features=25)
        self.fc_frame_att = paddle.nn.Linear(in_features=512, out_features=25)
        self.fc_av_att = paddle.nn.Linear(in_features=512, out_features=25)
        self.fc_a = paddle.nn.Linear(in_features=128, out_features=512)
        self.fc_v = paddle.nn.Linear(in_features=2048, out_features=512)
        self.fc_st = paddle.nn.Linear(in_features=512, out_features=512)
        self.fc_fusion = paddle.nn.Linear(in_features=1024, out_features=512)
        self.fc_occ_class_a = paddle.nn.Linear(in_features=512,
            out_features=25 * opt.occ_dim)
        self.fc_occ_class_v = paddle.nn.Linear(in_features=512,
            out_features=25 * opt.occ_dim)
        self.fc_occ_class_av = paddle.nn.Linear(in_features=512 * 2,
            out_features=25 * opt.occ_dim)
        self.fc_occ_v_q = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=opt.occ_dim)
        self.fc_occ_v_k = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=opt.occ_dim)
        self.fc_occ_v_v = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=opt.occ_dim)
        self.fc_occ_a_q = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=opt.occ_dim)
        self.fc_occ_a_k = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=opt.occ_dim)
        self.fc_occ_a_v = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=opt.occ_dim)
        self.fc_occ_av_q = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=opt.occ_dim)
        self.fc_occ_av_k = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=opt.occ_dim)
        self.fc_occ_av_v = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=opt.occ_dim)
        self.fc_occ_prob_a = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=1)
        self.fc_occ_prob_v = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=1)
        self.fc_occ_frame_prob = paddle.nn.Linear(in_features=opt.occ_dim,
            out_features=1)
        self.fc_occ_modality_prob = paddle.nn.Linear(in_features=opt.
            occ_dim, out_features=1)
        self.hat_encoder = Encoder(HANLayer(d_model=512, nhead=1,
            dim_feedforward=opt.forward_dim, dropout=opt.prob_drop),
            num_layers=1)
        self.av_class_encoder = Encoder(HANLayer(d_model=opt.occ_dim, nhead
            =1, dim_feedforward=opt.occ_dim, dropout=opt.prob_drop_occ),
            num_layers=1)
        self.frame_att_encoder = Encoder(HANLayer(d_model=opt.occ_dim,
            nhead=1, dim_feedforward=opt.occ_dim, dropout=opt.prob_drop_occ
            ), num_layers=1)
        self.modality_encoder = Encoder(HANLayer(d_model=opt.occ_dim, nhead
            =1, dim_feedforward=opt.occ_dim, dropout=opt.prob_drop_occ),
            num_layers=1)

    def co_occurrence(self, input, opt):
        x1_class, x2_class = input
        q_a = self.fc_occ_a_q(x1_class)
        k_a = self.fc_occ_a_k(x1_class)
        v_a = self.fc_occ_a_v(x1_class)
        q_v = self.fc_occ_v_q(x2_class)
        k_v = self.fc_occ_v_k(x2_class)
        v_v = self.fc_occ_v_v(x2_class)
        q_av = paddle.concat(x=(q_a, q_v), axis=1)
        k_av = paddle.concat(x=(k_a, k_v), axis=1)
        v_av = paddle.concat(x=(v_a, v_v), axis=1)
        att_1 = paddle.nn.functional.softmax(x=paddle.bmm(x=q_av, y=k_av.
            transpose(perm=[0, 2, 1])) / np.sqrt(opt.occ_dim), axis=-1)
        res_av = paddle.bmm(x=att_1, y=v_av).view(x1_class.shape[0], 50, -1)
        return res_av

    def forward(self, audio, visual, visual_st, opt, a_refine=None,
        v_refine=None, rand_idx=None, sample_idx=None, label=None, target=None
        ):
        x1_audio = self.fc_a(audio)
        vid_s = self.fc_v(visual).transpose(perm=[0, 2, 1]).unsqueeze(axis=-1)
        vid_s = paddle.nn.functional.avg_pool2d(kernel_size=(8, 1), x=vid_s,
            exclusive=False).squeeze(axis=-1).transpose(perm=[0, 2, 1])
        vid_st = self.fc_st(visual_st)
        x2 = paddle.concat(x=(vid_s, vid_st), axis=-1)
        x2_visual = self.fc_fusion(x2)
        x1, x2 = self.hat_encoder(x1_audio, x2_visual)

        # 应用全连接层和ReLU激活函数，然后调整张量形状
        x1_class = self.fc_occ_class_a(x1)
        x1_class = F.relu(x1_class)
        x1_class = paddle.reshape(x1_class, [x1.shape[0] * 10, 25, opt.occ_dim])

        x2_class = self.fc_occ_class_v(x2)
        x2_class = F.relu(x2_class)
        x2_class = paddle.reshape(x2_class, [x2.shape[0] * 10, 25, opt.occ_dim])

        # 使用你的自定义模块处理类属性特征
        x1_class_att, x2_class_att = self.av_class_encoder(x1_class, x2_class)

        # 应用全连接层，激活函数sigmoid，并调整张量形状
        x1_class_att = self.fc_occ_prob_a(x1_class_att)
        x1_class_att = F.sigmoid(x1_class_att)
        x1_class_att = paddle.reshape(x1_class_att, [x1.shape[0], 10, 1, -1])

        x2_class_att = self.fc_occ_prob_v(x2_class_att)
        x2_class_att = F.sigmoid(x2_class_att)
        x2_class_att = paddle.reshape(x2_class_att, [x2.shape[0], 10, 1, -1])

        frame_prob = paddle.concat(x=(x1_class_att, x2_class_att), axis=-2)
        x = paddle.concat(x=[x1.unsqueeze(axis=-2), x2.unsqueeze(axis=-2)],
            axis=-2)
        frame_att = paddle.nn.functional.softmax(x=self.fc_frame_att(x), axis=1
            )
        av_att = paddle.nn.functional.softmax(x=self.fc_av_att(x), axis=2)
        temporal_prob = frame_att * frame_prob
        global_prob = (temporal_prob * av_att).sum(axis=2).sum(axis=1)
        a_prob = temporal_prob[:, :, 0, :].sum(axis=1)
        v_prob = temporal_prob[:, :, 1, :].sum(axis=1)
        if rand_idx is not None:
            gg_weight = self.fc_frame_att(x)
            x1_target = a_refine.astype(dtype='float32')
            x2_target = v_refine.astype(dtype='float32')
            a_related_w = gg_weight[:, :, 0, :] * x1_target.unsqueeze(axis=1)
            a_related_w = a_related_w.sum(axis=-1).unsqueeze(axis=-1)
            a_related_w = paddle.nn.functional.softmax(x=a_related_w, axis=1)
            v_related_w = gg_weight[:, :, 1, :] * x2_target.unsqueeze(axis=1)
            v_related_w = v_related_w.sum(axis=-1).unsqueeze(axis=-1)
            v_related_w = paddle.nn.functional.softmax(x=v_related_w, axis=1)
            if opt.is_a_ori:
                a_agg = paddle.bmm(x=x1_audio.transpose(perm=[0, 2, 1]), y=
                    a_related_w)
            else:
                a_agg = paddle.bmm(x=x1.transpose(perm=[0, 2, 1]), y=
                    a_related_w)
            if opt.is_v_ori:
                v_agg = paddle.bmm(x=x2_visual.transpose(perm=[0, 2, 1]), y
                    =v_related_w)
            else:
                v_agg = paddle.bmm(x=x2.transpose(perm=[0, 2, 1]), y=
                    v_related_w)
            xx1 = paddle.nn.functional.normalize(x=a_agg, p=2, axis=1)
            xx2 = paddle.nn.functional.normalize(x=v_agg, p=2, axis=1)
            sims = paddle.mm(input=xx1.squeeze(axis=-1), mat2=xx2.squeeze(
                axis=-1).transpose(perm=[1, 0])) / opt.tmp
            return global_prob, a_prob, v_prob, frame_prob, sims
        return global_prob, a_prob, v_prob, frame_prob, (x1, x2)


class CMTLayer(paddle.nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(CMTLayer, self).__init__()
        self.self_attn = paddle.nn.MultiHeadAttention(d_model, nhead,
            dropout=dropout)
        self.linear1 = paddle.nn.Linear(in_features=d_model, out_features=
            dim_feedforward)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.linear2 = paddle.nn.Linear(in_features=dim_feedforward,
            out_features=d_model)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = paddle.nn.Dropout(p=dropout)
        self.dropout2 = paddle.nn.Dropout(p=dropout)
        self.activation = paddle.nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.

		Args:
			src: the sequnce to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
        src2 = self.self_attn(src_q, src_v, src_v, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout1(src2)
        src_q = self.norm1(src_q)
        src2 = self.linear2(self.dropout(paddle.nn.functional.relu(x=self.
            linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q
