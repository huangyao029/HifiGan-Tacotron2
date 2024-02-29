import json

from hparams_v1 import hparams_v1


hpv1 = hparams_v1()

chars = set()

with open(hpv1.summary_file, 'r', encoding = 'utf-8') as f:
    for line in f:
        words = line.strip().split('|')[-1].strip().split(' ')
        for w in words:
            chars.add(w)
            
chars = list(chars)
chars.sort()

_pad = '_'
_eos = '~'

symbols = [_pad, _eos] + chars

symbol_to_id_dict = {s : i for i, s in enumerate(symbols)}
id_to_symbol_dict = {i : s for i, s in enumerate(symbols)}

with open(hpv1.symbol_to_id, 'w') as f_s2i, open(hpv1.id_to_symbol, 'w') as f_i2s:
    json.dump(symbol_to_id_dict, f_s2i, indent = 4)
    json.dump(id_to_symbol_dict, f_i2s, indent = 4)
