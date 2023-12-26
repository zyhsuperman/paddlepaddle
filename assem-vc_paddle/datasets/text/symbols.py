""" from https://github.com/keithito/tacotron """
"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """
from . import cmudict
_pad = '<pad>'
_eos = '</s>'
_sos = '<s>'
_punc = "!'(),-.:~? "
_eng_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_arpabet = [('@' + s) for s in cmudict.valid_symbols]
_jamo_leads = ''.join([chr(_) for _ in range(4352, 4371)])
_jamo_vowels = ''.join([chr(_) for _ in range(4449, 4470)])
_jamo_tails = ''.join([chr(_) for _ in range(4520, 4547)])
_kor_characters = _jamo_leads + _jamo_vowels + _jamo_tails
_cht_characters = 'abcdefghijklmnopqrstuvwxyz12345'
_cmu_characters = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N',
    'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W',
    'Y', 'Z', 'ZH']
_cmu_characters = [('@' + s) for s in _cmu_characters]
eng_symbols = [_pad, _eos] + list(_eng_characters) + list(_punc) + _arpabet + [
    _sos]
cmu_symbols = [_pad, _eos] + list(_eng_characters) + list(_punc
    ) + _cmu_characters + [_sos]
kor_symbols = [_pad, _eos] + list(_kor_characters) + list(_punc) + [_sos]
cht_symbols = [_pad, _eos] + list(_cht_characters) + list(_punc) + [_sos]
