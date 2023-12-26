class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])

    Credits to epool:
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


hparams = Map({'name': 'wavenet_vocoder', 'builder': 'wavenet',
    'input_type': 'raw', 'quantize_channels': 65536, 'sample_rate': 16000,
    'silence_threshold': 2, 'num_mels': 80, 'fmin': 125, 'fmax': 7600,
    'fft_size': 1024, 'hop_size': 256, 'frame_shift_ms': None,
    'min_level_db': -100, 'ref_level_db': 20, 'rescaling': True,
    'rescaling_max': 0.999, 'allow_clipping_in_normalization': True,
    'log_scale_min': float(-32.23619130191664), 'out_channels': 10 * 3,
    'layers': 24, 'stacks': 4, 'residual_channels': 512, 'gate_channels': 
    512, 'skip_out_channels': 256, 'dropout': 1 - 0.95, 'kernel_size': 3,
    'weight_normalization': True, 'legacy': True, 'cin_channels': 80,
    'upsample_conditional_features': True, 'upsample_scales': [4, 4, 4, 4],
    'freq_axis_kernel_size': 3, 'gin_channels': -1, 'n_speakers': -1,
    'pin_memory': True, 'num_workers': 2, 'test_size': 0.0441,
    'test_num_samples': None, 'random_state': 1234, 'batch_size': 2,
    'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_eps': 1e-08, 'amsgrad': 
    False, 'initial_learning_rate': 0.001, 'lr_schedule':
    'noam_learning_rate_decay', 'lr_schedule_kwargs': {}, 'nepochs': 2000,
    'weight_decay': 0.0, 'clip_thresh': -1, 'max_time_sec': None,
    'max_time_steps': 8000, 'exponential_moving_average': True, 'ema_decay':
    0.9999, 'checkpoint_interval': 10000, 'train_eval_interval': 10000,
    'test_eval_epoch_interval': 5, 'save_optimizer_state': True})


def hparams_debug_string():
    values = hparams.values()
    hp = [('  %s: %s' % (name, values[name])) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
