import torch
import os, glob, librosa, tqdm, random
import numpy as np

from torch.utils.data import Dataset, DataLoader


class Tacotron2_Dataset(Dataset):
    def __init__(self, para):
        self.file_scp = para.train_scp
        files = np.loadtxt(self.file_scp, dtype = 'str', delimiter = '|')
        self.file_ids = files[:, 0].tolist()
        self.index_phone = files[:, 2].tolist()
        self.para = para
        
    # 读取特征
    def get_mel(self, file_id):
        file_fea = os.path.join(self.para.path_fea, file_id + '.npy')
        melspec = torch.from_numpy(np.load(file_fea, allow_pickle = True).astype(float)).to(torch.float32)
        return melspec
    
    # 读取文本编码序列
    def get_text(self, str_phones):
        phone_ids = [int(id) for id in str_phones.split()]
        return torch.IntTensor(phone_ids)
    
    # 获取 文本-特征 对
    def get_mel_text_pair(self, file_id, str_phones_ids):
        return (self.get_text(str_phones_ids), self.get_mel(file_id))
    
    def __getitem__(self, index):
        return self.get_mel_text_pair(self.file_ids[index], self.index_phone[index])
    
    def __len__(self):
        return len(self.file_ids)
    
    
class TextMelCollate():
    '''
    通过补0的方法使一个batch内的text和mel一样长，
    对mel进行补0的时候要让最长的mel时frames per step的整数倍
    '''
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step
        
    def __call__(self, batch):
        # text相关
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim = 0,
            descending = True
        )
        max_input_len = input_lengths[0]
        
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            
        # mel相关
        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0
        
        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)
        
        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths