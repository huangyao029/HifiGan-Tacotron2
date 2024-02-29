import logging
import torch
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from hparams import hparams
from datasets import Tacotron2_Dataset, TextMelCollate
from model import Tacotron2
from math import sqrt

def adjust_lr_rate(optimizer, lr, gamma, lr_final):
    lr_new = max(lr * gamma, lr_final)
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr_new
    return lr_new, optimizer

if __name__ == '__main__':
    file_log = './Tacotron2.log'
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        handlers = [
            logging.FileHandler(file_log),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    device = torch.device('cuda')
    
    para = hparams()
    
    model = Tacotron2(para)
    model.to(device)
    
    lr = para.lr
    optimizer = torch.optim.Adam(model.parameters(), lr, [0.9, 0.999], weight_decay = para.weight_decay)
    
    fun_loss_mel_out = torch.nn.MSELoss(reduction = 'sum')
    fun_loss_mel_posnet_out = torch.nn.MSELoss(reduction = 'sum')
    fun_loss_gate = torch.nn.BCEWithLogitsLoss()
    
    dataset = Tacotron2_Dataset(para)
    collate_fn = TextMelCollate(para.n_frames_per_step)
    dataloader = DataLoader(dataset, batch_size = para.batch_size, shuffle = True, num_workers = 8, collate_fn = collate_fn)
    
    n_step = 0
    
    model.train()
    for epoch in range(para.n_epoch):
        for i, batch_samples in enumerate(dataloader):
            
            time.sleep(1)
            
            n_step += 1
            
            text_in = batch_samples[0].to(device)
            text_lengths = batch_samples[1].to(device)
            
            target_mel = batch_samples[2].to(device)
            mel_lengths = batch_samples[4].to(device)
            
            target_gate = batch_samples[3].to(device)
            
            total_size = torch.sum(mel_lengths) * para.n_mel_channels
            
            model.zero_grad()
            mel_outputs, mel_outputs_postnet, gate_outputs, _ = model(text_in, text_lengths, target_mel, mel_lengths)
            
            loss_mel_out = fun_loss_mel_out(mel_outputs, target_mel)
            loss_mel_out_postnet = fun_loss_mel_posnet_out(mel_outputs_postnet, target_mel)
            
            target_gate = target_gate.view(-1, 1)
            gate_outputs = gate_outputs.view(-1, 1)
            loss_gate = fun_loss_gate(gate_outputs, target_gate)
            
            loss_all = loss_mel_out / total_size + loss_mel_out_postnet / total_size + loss_gate
            
            loss_all.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), para.grad_clip_thresh)
            
            optimizer.step()
            
            if n_step > para.start_lr_decay:
                lr, optimizer = adjust_lr_rate(optimizer, lr, para.gamma, para.lr_final)
                
            logger.info('epoch = %04d step %8d loss_all = %f loss_mse = %f loss_bce = %f'%(
                epoch, n_step, loss_all, loss_mel_out / total_size + loss_mel_out_postnet / total_size, loss_gate))
            
            if n_step % (para.step_save) == 0:
                path_save = os.path.join(para.path_save, str(n_step))
                os.makedirs(path_save, exist_ok = True)
                torch.save(
                    {
                        'model' : model.state_dict(),
                        'opt' : optimizer.state_dict()
                    },
                    os.path.join(path_save, 'model.pick')
                )
            
            