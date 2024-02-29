import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import glob
import tqdm
import time
import json

import soundfile as sf
import numpy as np

from model import Generator
from hparams import Hparams
from utils import load_checkpoint, scan_checkpoint
from preprocessor import wav2melspec

from Tacotron2.model import Tacotron2
from Tacotron2.parser_text_to_pyin import get_pyin
from Tacotron2.hparams import hparams as hparams_taco2


def generate_text_code(text_py, para : hparams_taco2):
    
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


def inference():
    
    hp = Hparams()
    device = None
    
    torch.manual_seed(hp.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    generator = Generator(hp).to(device)
    
    state_dict_g = load_checkpoint(hp.infer_checkpoint, device)
    generator.load_state_dict(state_dict_g['generator'])
    
    filelist = glob.glob(os.path.join(hp.in_test_wave_dir, '*.wav'))
    
    os.makedirs(hp.out_test_wave_dir, exist_ok = True)
    
    feature_mean, feature_std = np.load(hp.static_mean_std, allow_pickle = True)
    
    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filename in tqdm.tqdm(enumerate(filelist)):
            wav, sr = sf.read(filename)
            if sr != hp.sample_rate:
                wav = librosa.resample(wav, orig_sr = sr, target_sr = hp.sample_rate)
            mel = wav2melspec(wav, feature_mean, feature_std, hp)
            mel = torch.from_numpy(mel.astype(float)).to(torch.float32).unsqueeze(0).to(device)
            y_g_hat = generator(mel)
            audio = y_g_hat.squeeze()
            audio = audio.cpu().numpy().astype('float')
            
            output_file = os.path.join(hp.out_test_wave_dir, 
                                       '%s_%s_generated.wav'%(os.path.basename(filename).split('.wav')[0],
                                                              os.path.basename(hp.infer_checkpoint).split('_')[-1]))
            sf.write(output_file, audio, hp.sample_rate)
            
            
def inference_from_mel(mel_arr):
    
    assert len(mel_arr.shape) == 3, "mel_arr shape should be [N_item, N_mel_channel, N_frame]"
    
    hp = Hparams()
    device = None
    
    torch.manual_seed(hp.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    generator = Generator(hp).to(device)
    
    state_dict_g = load_checkpoint(hp.infer_checkpoint, device)
    generator.load_state_dict(state_dict_g['generator'])
    
    os.makedirs(hp.out_test_wave_feature_dir, exist_ok = True)
    
    feature_mean, feature_std = np.load(hp.static_mean_std, allow_pickle = True)
    
    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, mel in tqdm.tqdm(enumerate(mel_arr)):
            mel = (mel - feature_mean) / feature_std
            mel = torch.from_numpy(mel.astype(float)).to(torch.float32).unsqueeze(0).to(device)
            y_g_hat = generator(mel)
            audio = y_g_hat.squeeze()
            audio = audio.cpu().numpy().astype('float')
            output_file = os.path.join(hp.out_test_wave_feature_dir, 
                                       '%04d_generated_from_Tacotron2.wav'%(i))
            sf.write(output_file, audio, hp.sample_rate)
            
            
def inference_from_text(text_cn_list):
    
    hp_vocoder = Hparams()
    hp_acoustic = hparams_taco2()
    
    device = None
    
    torch.manual_seed(hp_vocoder.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp_vocoder.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    time_start_load_model = time.time()
    print('#### Loading model...')
    
    vocoder_generator = Generator(hp_vocoder).to(device)
    ascoustic_tacotron2 = Tacotron2(hp_acoustic).to(device)
    
    state_dict_g = load_checkpoint(hp_vocoder.infer_checkpoint, device)
    vocoder_generator.load_state_dict(state_dict_g['generator'])
    
    n_step_model = 48000
    state_dict_taco2 = torch.load(os.path.join(hp_acoustic.path_save, str(n_step_model), 'model.pick'))
    ascoustic_tacotron2.load_state_dict(state_dict_taco2['model'])
    
    time_end_load_model = time.time()
    
    print('#### Time taken to load the model : %.2f ms'%((time_end_load_model - time_start_load_model) * 1000))
    
    os.makedirs(hp_vocoder.out_test_wave_dir, exist_ok = True)
    
    vocoder_generator.eval()
    ascoustic_tacotron2.eval()
    vocoder_generator.remove_weight_norm()
    
    
    with torch.no_grad():
        for i, text_cn in enumerate(text_cn_list):
            
            print()
            print('## id = %04d'%(i))
            
            time_start_code_text = time.time()
            
            text_py = get_pyin(text_cn)[0]
            coded_text = generate_text_code(text_py, hp_acoustic)
            coded_text = np.array(coded_text)
            text_in = torch.from_numpy(coded_text)
            text_in = text_in.unsqueeze(0).to(device)
            
            time_end_code_text = time.time()
            print('## Time taken to code the text : %.2f ms'%((time_end_code_text - time_start_code_text) * 1000))
            
            taco2_output = ascoustic_tacotron2.inference(text_in)
            mel_output = taco2_output[1]
            # mel_output = mel_output.squeeze(0)
            # mel_output = mel_output.cpu().detach().numpy()
            
            time_end_taco2_infer = time.time()
            print('## Time taken to infer by ascoustic model : %.2f ms'%((time_end_taco2_infer - time_end_code_text) * 1000))
            
            y_g_hat = vocoder_generator(mel_output)
            
            time_end_vocoder_infer = time.time()
            print('## Time taken to infer by vocoder : %.2f ms'%((time_end_vocoder_infer - time_end_taco2_infer) * 1000))
            
            audio = y_g_hat.squeeze()
            audio = audio.cpu().numpy().astype('float')
            
            output_file = os.path.join(hp_vocoder.out_test_wave_feature_dir, 
                                       '%04d_%s_generated_from_Tacotron2.wav'%(i, os.path.basename(hp_vocoder.infer_checkpoint)))
            sf.write(output_file, audio, hp_vocoder.sample_rate)
            
            time_end_save_wave = time.time()
            print('## Time taken to save wave file : %.2f ms'%((time_end_save_wave - time_end_vocoder_infer) * 1000))
            print('## The length of wave : %.2f ms'%(len(audio) / hp_vocoder.sample_rate * 1000))
    
    
    
            
if __name__ == '__main__':
    # inference()
    # mel = np.load('/ZFS4T/tts/data/Hifi_GAN/Tacotron2/test_save/48000/out.mel.npy', allow_pickle = True)
    # n_channel, n_frame = mel.shape
    # mel_arr = mel.reshape(1, n_channel, n_frame)
    # inference_from_mel(mel_arr)
    
    text_cn_list = [
        '一个正在路上行走的银行家，觉得这个行当不行。',
        '卡尔普陪外孙玩滑梯。',
        '假语村言别再拥抱我。',
        '宝马配挂跛骡鞍，貂蝉怨枕董翁榻。',
        '邓小平与撒切尔会晤。',
        '五月天玛莎替菊娃娃换装比基尼。',
        '在一个迷人的森林中，有一只小松鼠名叫小橙，他梦想着成为一名勇敢的探险家。有一天，他踏上了旅程，遇到了一只友好的小兔子，他们一起经历了惊险刺激的冒险，最终实现了自己的梦想。'
    ]
    inference_from_text(text_cn_list)