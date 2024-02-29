import torch
import time
import os
import json
import librosa
import math

import numpy as np
import soundfile as sf

from model import Tacotron2
from hparams import hparams
from parser_text_to_pyin import get_pyin


def generate_text_code(text_py, para : hparams):
    
    if type(text_py) == str:
        text_py = text_py.split(' ')
        
    with open(para.symbol_to_id, 'r') as f_s2i:
        symbol_to_id_dict = json.load(f_s2i)
        
    sequence = []
    for w in text_py:
        if w in symbol_to_id_dict:
            sequence.append(symbol_to_id_dict[w])
        else:
            raise ValueError('not found %s in symbol_to_id_dict'%(w))
        
    sequence.append(symbol_to_id_dict['~'])
    
    return sequence


def text2speech(model : Tacotron2, para : hparams, coded_text_in, device):
    
    text_in = torch.from_numpy(coded_text_in)
    text_in = text_in.unsqueeze(0).to(device)
    
    # 解码
    start_infer = time.time()
    with torch.no_grad():
        eval_outputs = model.inference(text_in)
        mel_out = eval_outputs[1]
        mel_out = mel_out.squeeze(0)
        mel_out = mel_out.cpu().detach().numpy()
    end_infer = time.time()
        
    feature_mean, feature_std = np.load(para.static_file, allow_pickle = True)
    feature_mean = np.float64(feature_mean)
    feature_std = np.float64(feature_std)
    
    # 反正则
    generated_mel = mel_out * feature_std + feature_mean
    
    inv_fbank = librosa.db_to_power(generated_mel)
    
    inv_wav_lst = []
    max_n_frame_parse = 128
    n_parse = math.ceil(inv_fbank.shape[1] / max_n_frame_parse)
    for i in range(n_parse):
        inv_wav = librosa.feature.inverse.mel_to_audio(
            inv_fbank[:, i * max_n_frame_parse : (i + 1) * max_n_frame_parse],
            sr = para.fs,
            n_fft = para.n_fft,
            win_length = para.win_length,
            hop_length = para.hop_length,
            fmin = para.fmin,
            fmax = para.fmax
        )
        inv_wav_lst.append(inv_wav)
        
    return np.concatenate(inv_wav_lst), generated_mel, end_infer - start_infer
    


if __name__ == '__main__':
    
    para = hparams()
    
    device = torch.device('cuda')
    n_step_model = 48000
    
    time_start = time.time()
    
    model_loaded = torch.load(os.path.join(para.path_save, str(n_step_model), 'model.pick'))
    
    model = Tacotron2(para)
    model.to(device)
    model.load_state_dict(model_loaded['model'])
    
    time_load_model = time.time()
    
    model.eval()
    
    test_save_dir = os.path.join(para.path_save_test, str(n_step_model))
    os.makedirs(test_save_dir, exist_ok = True)
    
    time_start_infer = time.time()
    
    # 输入的文字
    text = '一个正在路上行走的银行家，觉得这个行当不行。'
    text_py = get_pyin(text)[0]
    coded_text = generate_text_code(text_py, para)
    
    time_text_code = time.time()
    
    wav, mel, infer_time = text2speech(model, para, np.array(coded_text), device)
    
    time_end_infer = time.time()
    
    sf.write(os.path.join(test_save_dir, 'out.wav'), wav, para.fs)
    np.save(os.path.join(test_save_dir, 'out.mel.npy'), mel)
    
    
    print('*' * 100)
    print('【原文】%s\n【拼音】%s'%(text, text_py))
    print('*' * 100)
    print('【模型加载耗时】%.2f ms'%((time_load_model - time_start) * 1000))
    print('【文本编码耗时】%.2f ms'%((time_text_code - time_start_infer) * 1000))
    print('【模型推理耗时】%.2f ms'%(infer_time * 1000))
    print('【mel2wav耗时】%.2f ms'%(((time_end_infer - time_text_code) - infer_time) * 1000))
    print('*' * 100)