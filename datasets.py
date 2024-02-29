import torch
import glob
import random
import os
import math
import copy

import numpy as np
import soundfile as sf

from hparams import Hparams
from preprocessor import wav2melspec


class MelDataset(torch.utils.data.Dataset):
    
    def __init__(self, hp : Hparams, shuffle = True, split = True, is_train = True, is_finetune = False):
        
        self.is_train = is_train
        self.is_finetune = is_finetune
        self.ids = []
        
        if is_train and not is_finetune:
            for pth in glob.glob(os.path.join(hp.out_feature_dir, '*.npy')):
                self.ids.append(os.path.basename(pth).split('.npy')[0])
        elif is_train and is_finetune:
            for pth in glob.glob(os.path.join(hp.finetune_feature_tr_dir, '*_gta.npy')):
                self.ids.append(os.path.basename(pth).split('.npy')[0])
        elif not is_train and not is_finetune:
            for pth in glob.glob(os.path.join(hp.out_feature_cv_dir, '*.npy')):
                self.ids.append(os.path.basename(pth).split('.npy')[0])
        elif not is_train and is_finetune:
            for pth in glob.glob(os.path.join(hp.finetune_feature_cv_dir, '*_gta.npy')):
                self.ids.append(os.path.basename(pth).split('.npy')[0])
                
        print('data number : ', len(self.ids))
                
        random.seed(123)
        if shuffle and is_train:
            random.shuffle(self.ids)
        self.shuffle = shuffle
        self.split = split
        self.hp = hp
        
        self.feature_mean, self.feature_std = np.load(hp.static_mean_std, allow_pickle = True)
       
        
    def __getitem__(self, index):
        
        id_str = self.ids[index]
        
        # if self.is_train:
        #     if id_str[-3:] == 'gta':
        #         audio_file = os.path.join(self.hp.finetune_wave_dir, '%s.wav'%(id_str.split('_gta')[0]))
        #     else:
        #         audio_file = os.path.join(self.hp.finetune_wave_dir, '%s.wav'%(id_str))
        #     #feature_file = os.path.join(self.hp.out_feature_dir, '%s.npy'%(id_str))
        # else:
        #     audio_file = os.path.join(self.hp.out_wave_cv_dir, '%s.wav'%(id_str))
            
            
        if id_str[-4:] == '_gta':
            audio_file = os.path.join(self.hp.finetune_wave_dir, '%s.wav'%(id_str.split('_gta')[0]))
        else:
            audio_file = os.path.join(self.hp.finetune_wave_dir, '%s.wav'%(id_str))

        audio = sf.read(audio_file)[0]
        
        if not self.is_finetune:
            if self.split:
                if audio.shape[0] >= self.hp.segment_size:
                    max_audio_start = audio.shape[0] - self.hp.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[audio_start : audio_start + self.hp.segment_size]
                else:
                    audio = np.pad(audio, (0, self.hp.segment_size - audio.shape[0]), 'constant')
                    
            mel = wav2melspec(audio, self.feature_mean, self.feature_std, self.hp)
        else:
            if self.is_train:
                mel = np.load(os.path.join(self.hp.finetune_feature_tr_dir, '%s.npy'%(id_str)), allow_pickle = True)
            else:
                mel = np.load(os.path.join(self.hp.finetune_feature_cv_dir, '%s.npy'%(id_str)), allow_pickle = True)
            # if len(mel.shape) == 2:
            #     mel = mel.reshape(1, mel.shape[0], mel.shape[1])
            if self.split:
                frames_per_seg = math.ceil(self.hp.segment_size / self.hp.hop_size)
                if audio.shape[0] >= self.hp.segment_size:
                    mel_start = random.randint(0, mel.shape[1] - frames_per_seg - 1)
                    mel = mel[:, mel_start : mel_start + frames_per_seg]
                    audio = audio[mel_start * self.hp.hop_size : (mel_start + frames_per_seg) * self.hp.hop_size]
                else:
                    mel = np.pad(mel, (0, frames_per_seg - mel.shape[1]), 'constant')
                    audio = np.pad(audio, (0, self.hp.segment_size - audio.shape[0]), 'constant')
            
        
        #mel_loss = wav2melspec(audio, self.feature_mean, self.feature_std, self.hp)
        mel_loss = copy.deepcopy(mel)
        
        return (torch.from_numpy(mel.astype(float)).to(torch.float32), torch.from_numpy(audio.astype(float)).to(torch.float32), 
                torch.from_numpy(mel_loss.astype(float)).to(torch.float32))
    
    
    def __len__(self):
        return len(self.ids)
        

if __name__ == '__main__':
    hp = Hparams()
    obj = MelDataset(hp)
    mel, audio, mel_loss = obj.__getitem__(100)
    print(mel.shape)
    print(audio.shape)
    np.save('./mel.npy', mel.numpy())
    np.save('./audio.npy', audio.numpy())