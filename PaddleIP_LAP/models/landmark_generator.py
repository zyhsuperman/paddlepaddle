import paddle
import math


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis]//num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)
    
class PositionalEmbedding(paddle.nn.Layer):

    def __init__(self, d_model=512, max_len=512):
        super().__init__()
        pe = paddle.zeros(shape=[max_len, d_model]).astype(dtype='float32')
        pe.stop_gradient = True
        position = paddle.arange(start=0, end=max_len).astype(dtype='float32'
            ).unsqueeze(axis=1)
        div_term = (paddle.arange(start=0, end=d_model, step=2).astype(
            dtype='float32') * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = paddle.sin(x=position * div_term)
        pe[:, 1::2] = paddle.cos(x=position * div_term)
        pe = pe.unsqueeze(axis=0)
        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x):
        return self.pe[:, :x.shape[1]]


class Conv1d(paddle.nn.Layer):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=
        False, act='ReLU', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = paddle.nn.Sequential(paddle.nn.Conv1D(in_channels
            =cin, out_channels=cout, kernel_size=kernel_size, stride=stride,
            padding=padding), paddle.nn.BatchNorm1D(num_features=cout))
        if act == 'ReLU':
            self.act = paddle.nn.ReLU()
        elif act == 'Tanh':
            self.act = paddle.nn.Tanh()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2d(paddle.nn.Layer):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=
        False, act='ReLU', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels
            =cin, out_channels=cout, kernel_size=kernel_size, stride=stride,
            padding=padding), paddle.nn.BatchNorm2D(num_features=cout))
        if act == 'ReLU':
            self.act = paddle.nn.ReLU()
        elif act == 'Tanh':
            self.act = paddle.nn.Tanh()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


def weight_init(m):
    if isinstance(m, paddle.nn.Linear):
        init_XavierNormal = paddle.nn.initializer.XavierNormal()
        init_XavierNormal(m.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(m.bias)
    elif isinstance(m, paddle.nn.BatchNorm1D):
        init_Constant = paddle.nn.initializer.Constant(value=1)
        init_Constant(m.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(m.bias)


class Fusion_transformer_encoder(paddle.nn.Layer):

    def __init__(self, T, d_model, nlayers, nhead, dim_feedforward, dropout=0.1
        ):
        super().__init__()
        self.T = T
        self.position_v = PositionalEmbedding(d_model=512)
        self.position_a = PositionalEmbedding(d_model=512)
        self.modality = paddle.nn.Embedding(num_embeddings=4, embedding_dim
            =512, padding_idx=0)
        self.dropout = paddle.nn.Dropout(p=dropout)
        encoder_layers = paddle.nn.TransformerEncoderLayer(d_model, nhead,
            dim_feedforward, dropout)
        self.transformer_encoder = paddle.nn.TransformerEncoder(encoder_layers,
            nlayers)

    def forward(self, ref_embedding, mel_embedding, pose_embedding):
        position_v_encoding = self.position_v(pose_embedding)
        position_a_encoding = self.position_a(mel_embedding)
        modality_v = self.modality(1 * paddle.ones(shape=(pose_embedding.
            shape[0], self.T), dtype='int32'))
        modality_a = self.modality(2 * paddle.ones(shape=(mel_embedding.
            shape[0], self.T), dtype='int32'))
        pose_tokens = pose_embedding + position_v_encoding + modality_v
        audio_tokens = mel_embedding + position_a_encoding + modality_a
        ref_tokens = ref_embedding + self.modality(3 * paddle.ones(shape=(
            ref_embedding.shape[0], ref_embedding.shape[1]), dtype='int32'))
        input_tokens = paddle.concat(x=(ref_tokens, audio_tokens,
            pose_tokens), axis=1)
        input_tokens = self.dropout(input_tokens)
        output = self.transformer_encoder(input_tokens)
        return output


class Landmark_generator(paddle.nn.Layer):

    def __init__(self, T, d_model, nlayers, nhead, dim_feedforward, dropout=0.1
        ):
        super(Landmark_generator, self).__init__()
        self.mel_encoder = paddle.nn.Sequential(
            Conv2d(1, 32, kernel_size=3,stride=1, padding=1), 
            Conv2d(32, 32, kernel_size=3, stride=1,padding=1, residual=True), 

            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(32, 64, kernel_size=3,stride=(3, 1), padding=1),

            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(64, 64, kernel_size=3,stride=1, padding=1, residual=True), 

            Conv2d(64, 128,kernel_size=3, stride=3, padding=1), 
            Conv2d(128, 128,kernel_size=3, stride=1, padding=1, residual=True), 

            Conv2d(128,128, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1), 

            Conv2d(256,256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0), 
            Conv2d(512, 512,kernel_size=1, stride=1, padding=0, act='Tanh'))
        
        self.ref_encoder = paddle.nn.Sequential(
            Conv1d(2, 4, 3, 1, 1),
            Conv1d(4, 8, 3, 2, 1), 

            Conv1d(8, 8, 3, 1, 1, residual=True),
            Conv1d(8, 8, 3, 1, 1, residual=True), 

            Conv1d(8, 16, 3, 2, 1),
            Conv1d(16, 16, 3, 1, 1, residual=True), 
            Conv1d(16, 16, 3, 1, 1,residual=True), 

            Conv1d(16, 32, 3, 2, 1), 
            Conv1d(32, 32, 3, 1, 1,residual=True), 
            Conv1d(32, 32, 3, 1, 1, residual=True), 

            Conv1d(32, 64, 3, 2, 1), 
            Conv1d(64, 64, 3, 1, 1, residual=True),
            Conv1d(64, 64, 3, 1, 1, residual=True),

            Conv1d(64, 128, 3, 2, 1), 
            Conv1d(128, 128, 3, 1, 1, residual=True), 
            Conv1d(128, 128, 3,1, 1, residual=True), 

            Conv1d(128, 256, 3, 2, 1), 
            Conv1d(256, 256, 3, 1, 1, residual=True), 
            Conv1d(256, 512, 3, 1, 0), 
            Conv1d(512, 512, 1, 1, 0, act='Tanh'))
        self.pose_encoder = paddle.nn.Sequential(
            Conv1d(2, 4, 3, 1, 1),
            Conv1d(4, 8, 3, 1, 1), 

            Conv1d(8, 8, 3, 1, 1, residual=True),
            Conv1d(8, 8, 3, 1, 1, residual=True), 

            Conv1d(8, 16, 3, 2, 1),
            Conv1d(16, 16, 3, 1, 1, residual=True), 
            Conv1d(16, 16, 3, 1, 1,residual=True), 

            Conv1d(16, 32, 3, 2, 1), 
            Conv1d(32, 32, 3, 1, 1,residual=True), 
            Conv1d(32, 32, 3, 1, 1, residual=True),
            
            Conv1d(32, 64, 3, 2, 1), 
            Conv1d(64, 64, 3, 1, 1, residual=True),
            Conv1d(64, 64, 3, 1, 1, residual=True), 

            Conv1d(64, 128, 3, 2, 1), 
            Conv1d(128, 128, 3, 1, 1, residual=True), 
            Conv1d(128, 128, 3, 1, 1, residual=True), 

            Conv1d(128, 256, 3, 2, 1), 
            Conv1d(256, 256, 3, 1, 1, residual=True), 
            Conv1d(256, 256, 3, 1, 1,residual=True), 

            Conv1d(256, 512, 3, 1, 0), 
            Conv1d(512, 512, 1, 1, 0, residual=True, act='Tanh'))
        self.fusion_transformer = Fusion_transformer_encoder(T, d_model,
            nlayers, nhead, dim_feedforward, dropout)
        self.mouse_keypoint_map = paddle.nn.Linear(in_features=d_model,
            out_features=40 * 2)
        self.jaw_keypoint_map = paddle.nn.Linear(in_features=d_model,
            out_features=17 * 2)
        self.apply(weight_init)
        self.Norm = paddle.nn.LayerNorm(normalized_shape=512)

    def forward(self, T_mels, T_pose, Nl_pose, Nl_content):
        B, T, N_l = T_mels.shape[0], T_mels.shape[1], Nl_content.shape[1]
        Nl_ref = paddle.concat(x=[Nl_pose, Nl_content], axis=3)
        Nl_ref = paddle.concat(x=[Nl_ref[i] for i in range(Nl_ref.shape[0])
            ], axis=0)
        T_mels = paddle.concat(x=[T_mels[i] for i in range(T_mels.shape[0])
            ], axis=0)
        T_pose = paddle.concat(x=[T_pose[i] for i in range(T_pose.shape[0])
            ], axis=0)
        mel_embedding = self.mel_encoder(T_mels).squeeze(axis=-1).squeeze(axis
            =-1)
        pose_embedding = self.pose_encoder(T_pose).squeeze(axis=-1)
        ref_embedding = self.ref_encoder(Nl_ref).squeeze(axis=-1)
        mel_embedding = self.Norm(mel_embedding)
        pose_embedding = self.Norm(pose_embedding)
        ref_embedding = self.Norm(ref_embedding)
        mel_embedding = paddle.stack(x=split(x=mel_embedding,
            num_or_sections=T), axis=0)
        pose_embedding = paddle.stack(x=split(x=pose_embedding,
            num_or_sections=T), axis=0)
        ref_embedding = paddle.stack(x=split(x=ref_embedding,
            num_or_sections=N_l, axis=0), axis=0)
        output_tokens = self.fusion_transformer(ref_embedding,
            mel_embedding, pose_embedding)
        lip_embedding = output_tokens[:, N_l:N_l + T, :]
        jaw_embedding = output_tokens[:, N_l + T:, :]
        output_mouse_landmark = self.mouse_keypoint_map(lip_embedding)
        output_jaw_landmark = self.jaw_keypoint_map(jaw_embedding)
        predict_content = paddle.reshape(x=paddle.concat(x=[
            output_jaw_landmark, output_mouse_landmark], axis=2), shape=(B,
            T, -1, 2))
        predict_content = paddle.concat(x=[predict_content[i] for i in
            range(predict_content.shape[0])], axis=0).transpose(perm=[0, 2, 1])
        return predict_content
