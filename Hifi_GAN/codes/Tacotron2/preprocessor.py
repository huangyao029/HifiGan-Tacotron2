import librosa
import os
import tqdm
import glob
import torch

import numpy as np
import soundfile as sf

from hparams import Hparams


def wav_to_mel(wav, hp : Hparams):
    fbank = librosa.feature.melspectrogram(
        y = wav,
        sr = hp.sample_rate,
        n_fft = hp.n_fft,
        win_length = hp.win_size,
        hop_length = hp.hop_size,
        n_mels = hp.n_mels,
        fmin = hp.fmin,
        fmax = hp.fmax,
        center = False,
        pad_mode = 'reflect'
    )
    log_fbank = librosa.power_to_db(fbank)
    return log_fbank


def wav2melspec(wav, mean, std, hp : Hparams):
    if len(wav.shape) == 1:
        wav = np.pad(wav, (int((hp.n_fft - hp.hop_size) // 2), int((hp.n_fft - hp.hop_size) // 2)), mode = 'reflect')
    elif len(wav.shape) == 2:
        wav = np.pad(wav, ((0, 0), (int((hp.n_fft - hp.hop_size) // 2), int((hp.n_fft - hp.hop_size) // 2))), mode = 'reflect')
    elif len(wav.shape) == 3:
        wav = np.pad(wav, ((0, 0), (0, 0), (int((hp.n_fft - hp.hop_size) // 2), int((hp.n_fft - hp.hop_size) // 2))), mode = 'reflect')
    #return torch.from_numpy(((wav_to_mel(wav, hp) - mean) / std).astype(float))
    return (wav_to_mel(wav, hp) - mean) / std



# def wav2melspec_torch(y, hp : Hparams):
    
#     if torch.min(y) < -1.:
#         print('min value is : ', torch.min(y))
#     if torch.max(y) > 1.:
#         print('max value is : ', torch.max(y))

    
    


def mel_spectrogram(hp : Hparams):
    
    wav_file_lst = glob.glob(os.path.join(hp.in_wave_dir, '*.wav'))
    
    features = []
    ids = []
    
    print('Generating features...')
    
    os.makedirs(hp.out_feature_dir, exist_ok = True)
    os.makedirs(hp.out_wave_dir, exist_ok = True)
    
    for file_path in tqdm.tqdm(wav_file_lst):
        wav, sr = sf.read(file_path)
        if sr != hp.sample_rate:
            wav = librosa.resample(wav, orig_sr = sr, target_sr = hp.sample_rate)
            
        wav_id = os.path.basename(file_path).split('.wav')[0]
        
        out_wav_path = os.path.join(hp.out_wave_dir, '%s.wav'%(wav_id))
        sf.write(out_wav_path, wav, hp.sample_rate)
        
        feature = wav_to_mel(wav, hp)
        
        ids.append(wav_id)
        features.append(feature)
        
    # mean and std of features
    features_array = np.concatenate(features, axis = 1)
    features_mean = np.mean(features_array, axis = 1, keepdims = True)
    features_std = np.std(features_array, axis = 1, keepdims = True)
    
    print('Saving features...')
    for fea, wav_id, in tqdm.tqdm(zip(features, ids)):
        norm_fea = (fea - features_mean) / features_std
        out_feature_path = os.path.join(hp.out_feature_dir, '%s.npy'%(wav_id))
        np.save(out_feature_path, norm_fea)
    
    np.save(hp.static_mean_std, np.array([features_mean, features_std], dtype = object))
    
    
if __name__ == '__main__':
    hp = Hparams()
    mel_spectrogram(hp)
    
    # hp = Hparams()
    # mean, std = np.load(hp.static_mean_std, allow_pickle = True)
    # data = np.random.rand(8, 1, 8192)
    # out1 = wav2melspec(data, mean, std, hp)
    # out2 = wav2melspec(data[0, 0, :].reshape(-1), mean, std, hp)
    # out3 = wav2melspec(data[:, 0, :].reshape(8, -1), mean, std, hp)
    # print(out1.shape, out2.shape, out3.shape)
    # dif1 = out1[0, 0, :, :] - out2
    # dif2 = out3[0, :, :] - out2
    # print(np.mean(dif1))
    # print(np.mean(dif2))
    
    