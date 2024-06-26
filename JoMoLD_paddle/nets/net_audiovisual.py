import paddle


class LabelSmoothingNCELoss(paddle.nn.Layer):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingNCELoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = paddle.nn.functional.softmax(pred, axis=self.dim)
        with paddle.no_grad():
            true_dist = paddle.zeros_like(x=pred)
            true_dist.fill_(value=self.smoothing / (self.cls - 1))
            true_dist.put_along_axis_(axis=1, indices=target.data.unsqueeze
                (1), values=self.confidence)
        return -paddle.mean(x=paddle.log(x=paddle.sum(x=true_dist * pred,
            axis=self.dim)))


class Encoder(paddle.nn.Layer):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=512,
        nhead=1, dim_feedforward=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = []
        if encoder_layer == 'HANLayer':
            for i in range(num_layers):
                self.layers.append(HANLayer(d_model=d_model, nhead=nhead,
                    dim_feedforward=dim_feedforward, dropout=dropout))
        else:
            raise ValueError('wrong encoder layer')
        self.layers = paddle.nn.LayerList(sublayers=self.layers)
        self.num_layers = num_layers
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None,
        with_ca=True):
        output_a = src_a
        output_v = src_v
        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            src_a = output_a
            src_v = output_v
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

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=
        None, with_ca=True):
        """Pass the input through the encoder layer.

        Args:
            src_q: the sequence to the encoder layer (required).
            src_v: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            with_ca: whether to use audio-visual cross-attention
        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.transpose(perm=[1, 0, 2])
        src_v = src_v.transpose(perm=[1, 0, 2])
        if with_ca:
            src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask)[0]
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask)[0]
            src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
            src_q = self.norm1(src_q)
        else:
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask)[0]
            src_q = src_q + self.dropout12(src2)
            src_q = self.norm1(src_q)
        src_q = src_q + self.dropout2(self.linear2(self.dropout(paddle.nn.
            functional.relu(x=self.linear1(src_q)))))
        src_q = self.norm2(src_q)
        return src_q.transpose(perm=[1, 0, 2])


class MMIL_Net(paddle.nn.Layer):

    def __init__(self, num_layers=1, temperature=0.2, att_dropout=0.1,
        cls_dropout=0.5):
        super(MMIL_Net, self).__init__()
        self.fc_prob = paddle.nn.Linear(in_features=512, out_features=25)
        self.fc_frame_att = paddle.nn.Linear(in_features=512, out_features=25)
        self.fc_av_att = paddle.nn.Linear(in_features=512, out_features=25)
        self.fc_a = paddle.nn.Linear(in_features=128, out_features=512)
        self.fc_v = paddle.nn.Linear(in_features=2048, out_features=512)
        self.fc_st = paddle.nn.Linear(in_features=512, out_features=512)
        self.fc_fusion = paddle.nn.Linear(in_features=1024, out_features=512)
        self.hat_encoder = Encoder('HANLayer', num_layers, norm=None,
            d_model=512, nhead=1, dim_feedforward=512, dropout=att_dropout)
        self.temp = temperature
        if cls_dropout != 0:
            self.dropout = paddle.nn.Dropout(p=cls_dropout)
        else:
            self.dropout = None

    def forward(self, audio, visual, visual_st, with_ca=True):
        b, t, d = visual_st.shape
        x1 = self.fc_a(audio)
        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual).transpose((0, 2, 1)).unsqueeze(-1)
        vid_s = paddle.nn.functional.avg_pool2d(vid_s, (8, 1)).squeeze(-1).transpose((0, 2, 1))
        vid_st = self.fc_st(visual_st)
        x2 = paddle.concat([vid_s, vid_st], axis=-1)
        x2 = self.fc_fusion(x2)
        # HAN
        x1, x2 = self.hat_encoder(x1, x2, with_ca=with_ca)

        # noise contrastive
        xx2_after = paddle.nn.functional.normalize(x2, p=2, axis=-1)
        xx1_after = paddle.nn.functional.normalize(x1, p=2, axis=-1)
        sims_after = paddle.bmm(xx1_after, xx2_after.transpose((0, 2, 1))).squeeze(1) / self.temp
        sims_after = sims_after.reshape((-1, 10))
        mask_after = paddle.zeros([b, 10], dtype='int64')
        for i in range(10):
            mask_after[:, i] = i
        mask_after = mask_after.cuda()
        mask_after = mask_after.reshape((-1,))

        # prediction
        if self.dropout is not None:
            x2 = self.dropout(x2)
        x = paddle.concat([x1.unsqueeze(-2), x2.unsqueeze(-2)], axis=-2)
        frame_prob = paddle.nn.functional.sigmoid(self.fc_prob(x))
        # attentive MMIL pooling
        frame_att = paddle.nn.functional.softmax(self.fc_frame_att(x), axis=1)
        av_att = paddle.nn.functional.softmax(self.fc_av_att(x), axis=2)
        temporal_prob = frame_att * frame_prob
        global_prob = (temporal_prob * av_att).sum(axis=2).sum(axis=1)
        # frame-wise probability
        a_prob = temporal_prob[:, :, 0, :].sum(axis=1)
        v_prob = temporal_prob[:, :, 1, :].sum(axis=1)

        return global_prob, a_prob, v_prob, frame_prob, sims_after, mask_after