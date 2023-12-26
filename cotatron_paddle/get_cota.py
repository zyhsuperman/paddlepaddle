import paddle
ckpt = paddle.load(path=
    'checkpoints/pretrained_decoder_libritts_vctk_epoch652_15388cc.pdparams')
cotapath = 'checkpoints/cota.pdparams'
cota_state_dict = {}
for k, v in ckpt['state_dict'].items():
    if 'cotatron' in k:
        key = k.replace('cotatron.', '')
        cota_state_dict[key] = v
cota = {}
cota['state_dict'] = cota_state_dict
paddle.save(obj=cota, path=cotapath)
