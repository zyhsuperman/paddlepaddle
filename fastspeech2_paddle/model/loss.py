import paddle


class FastSpeech2Loss(paddle.nn.Layer):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config['preprocessing']['pitch'][
            'feature']
        self.energy_feature_level = preprocess_config['preprocessing']['energy'
            ]['feature']
        self.mse_loss = paddle.nn.MSELoss()
        self.mae_loss = paddle.nn.L1Loss()

    def forward(self, inputs, predictions):
        (mel_targets, _, _, pitch_targets, energy_targets, duration_targets
            ) = inputs[6:]
        (mel_predictions, postnet_mel_predictions, pitch_predictions,
            energy_predictions, log_duration_predictions, _, src_masks,
            mel_masks, _, _) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = paddle.log(x=duration_targets.astype(dtype=
            'float32') + 1)
        mel_targets = mel_targets[:, :mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        log_duration_targets.stop_gradient = not False
        pitch_targets.stop_gradient = not False
        energy_targets.stop_gradient = not False
        mel_targets.stop_gradient = not False
        if self.pitch_feature_level == 'phoneme_level':
            pitch_predictions = pitch_predictions.masked_select(mask=src_masks)
            pitch_targets = pitch_targets.masked_select(mask=src_masks)
        elif self.pitch_feature_level == 'frame_level':
            pitch_predictions = pitch_predictions.masked_select(mask=mel_masks)
            pitch_targets = pitch_targets.masked_select(mask=mel_masks)
        if self.energy_feature_level == 'phoneme_level':
            energy_predictions = energy_predictions.masked_select(mask=
                src_masks)
            energy_targets = energy_targets.masked_select(mask=src_masks)
        if self.energy_feature_level == 'frame_level':
            energy_predictions = energy_predictions.masked_select(mask=
                mel_masks)
            energy_targets = energy_targets.masked_select(mask=mel_masks)
        log_duration_predictions = log_duration_predictions.masked_select(mask
            =src_masks)
        log_duration_targets = log_duration_targets.masked_select(mask=
            src_masks)
        mel_predictions = mel_predictions.masked_select(mask=mel_masks.
            unsqueeze(axis=-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mask
            =mel_masks.unsqueeze(axis=-1))
        mel_targets = mel_targets.masked_select(mask=mel_masks.unsqueeze(
            axis=-1))
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions,
            log_duration_targets)
        total_loss = (mel_loss + postnet_mel_loss + duration_loss +
            pitch_loss + energy_loss)
        return (total_loss, mel_loss, postnet_mel_loss, pitch_loss,
            energy_loss, duration_loss)
