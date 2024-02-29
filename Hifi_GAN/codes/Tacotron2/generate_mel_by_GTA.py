import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import tqdm

import numpy as np
from torch.utils.data import DataLoader

from model import Tacotron2
from parser_text_to_pyin import get_pyin
from hparams import hparams
from datasets import Tacotron2_Dataset, TextMelCollate



def get_mel_by_gta():
    
    hp = hparams()
    device = None
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    n_step_model = 48000
        
    model = Tacotron2(hp).to(device)
    state_dict_model = torch.load(os.path.join(hp.path_save, str(n_step_model), 'model.pick'))
    model.load_state_dict(state_dict_model['model'])
    
    dataset = Tacotron2_Dataset(hp)
    collate_fn = TextMelCollate(hp.n_frames_per_step)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 1, collate_fn = collate_fn)
    
    os.makedirs(hp.gta_feature_dir, exist_ok = True)
    
    model.eval()
    with torch.no_grad():
        for i, sample in tqdm.tqdm(enumerate(dataloader)):
            
            text_in = sample[0].to(device)
            text_lengths = sample[1].to(device)
            
            target_mel = sample[2].to(device)
            mel_lengths = sample[4].to(device)
            
            target_gate = sample[3].to(device)
            
            total_size = torch.sum(mel_lengths) * hp.n_mel_channels
            
            mel_outputs, mel_outputs_postnet, gate_outputs, _ = model(text_in, text_lengths, target_mel, mel_lengths)
            
            mel_outputs_length = mel_outputs.size(2)

            assert mel_outputs_length >= mel_lengths[0]
            if not (mel_outputs_length - mel_lengths[0]) <= 2:
                print('mel_outputs_length - mel_length.size(0) = %d'%(mel_outputs_length - mel_lengths[0]))
                continue
            mel_outputs = mel_outputs[:, :, :mel_lengths[0]]
            
            mel_outputs_np = mel_outputs.squeeze(0).cpu().numpy()
            
            out_file_name = os.path.join(hp.gta_feature_dir, '%06d.npy'%(i+1))
            np.save(out_file_name, mel_outputs_np)
        
        
if __name__ == '__main__':
    get_mel_by_gta()
            
            
    