import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import time
import json
import itertools

import numpy as np
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.distributed import init_process_group
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel

from hparams import Hparams
from model import (Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, 
                   feature_loss, generator_loss, discriminator_loss)
from utils import load_checkpoint, save_checkpoint, scan_checkpoint
from datasets import MelDataset
from preprocessor import wav2melspec




def train(rank, hp : Hparams):
    
    if hp.num_gpus > 1:
        init_process_group(
            backend = hp.dist_backend, init_method = hp.dist_url,
            world_size = hp.world_size * hp.num_gpus, rank = rank)
    
    torch.cuda.manual_seed(hp.seed)
    device = torch.device('cuda:{:d}'.format(rank))
    
    generator = Generator(hp).to(device)
    mpd = MultiPeriodDiscriminator(hp).to(device)
    msd = MultiScaleDiscriminator(hp).to(device)
    
    if rank == 0:
        print(generator)
        os.makedirs(hp.checkpoint, exist_ok = True)
        print('checkpoint directory : ', hp.checkpoint)
        
    if os.path.isdir(hp.checkpoint):
        cp_g = scan_checkpoint(hp.checkpoint, 'g_')
        cp_do = scan_checkpoint(hp.checkpoint, 'do_')
        
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        
    if hp.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids = [rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids = [rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids = [rank]).to(device)
        
    optim_g = torch.optim.AdamW(generator.parameters(), hp.learning_rate, betas = [hp.adam_b1, hp.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), 
                                hp.learning_rate, betas = [hp.adam_b1, hp.adam_b2])
    
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
        
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma = hp.lr_decay, last_epoch = last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma = hp.lr_decay, last_epoch = last_epoch)
    
    trainset = MelDataset(hp, shuffle = False if hp.num_gpus > 1 else True, is_train = True, is_finetune = True)
    
    train_sampler = DistributedSampler(trainset) if hp.num_gpus > 1 else None
    
    train_loader = DataLoader(trainset, num_workers = hp.num_workers, shuffle = True, 
                              sampler = train_sampler, batch_size = hp.batch_size,
                              pin_memory = True, drop_last = True)
    
    if rank == 0:
        validset = MelDataset(hp, shuffle = False, is_train = False, split = True, is_finetune = True)
        validation_loader = DataLoader(validset, num_workers = 1, shuffle = False,
                                       sampler = None, batch_size = 1, 
                                       pin_memory = True, drop_last = True)
    
    feature_mean, feature_std = np.load(hp.static_mean_std, allow_pickle = True)
        
    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), hp.training_epochs):
        if rank == 0:
            start = time.time()
            print('Epoch : {}'.format(epoch + 1))
            
        if hp.num_gpus > 1:
            train_sampler.set_epoch(epoch)
            
        for i, batch in enumerate(train_loader):
            
            time.sleep(0)
            
            if rank == 0:
                start_b = time.time()
            x, y, y_mel = batch
            
            x = torch.autograd.Variable(x.to(device, non_blocking = True))
            y = torch.autograd.Variable(y.to(device, non_blocking = True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking = True))
            y = y.unsqueeze(1)
            
            y_g_hat = generator(x)
            y_g_hat_mel = wav2melspec(y_g_hat.detach().cpu().numpy(), feature_mean, feature_std, hp)
            y_g_hat_mel = torch.from_numpy(y_g_hat_mel.astype(float)).to(torch.float32).to(device, non_blocking = True)
            y_g_hat_mel = y_g_hat_mel.squeeze(1)
            
            optim_d.zero_grad()
            
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            
            loss_disc_all = loss_disc_s + loss_disc_f
            
            loss_disc_all.backward()
            optim_d.step()
            
            # Generator
            optim_g.zero_grad()
            
            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            
            loss_gen_all.backward()
            optim_g.step()
            
            if rank == 0:
                if steps % hp.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpoint
                if steps % hp.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = '{}/g_{:08d}'.format(hp.checkpoint, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator' : (generator.module if hp.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = '{}/do_{:08d}'.format(hp.checkpoint, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd' : (mpd.module if hp.num_gpus > 1 else mpd).state_dict(),
                                     'msd' : (msd.module if hp.num_gpus > 1 else msd).state_dict(),
                                     'optim_g' : optim_g.state_dict(),
                                     'optim_d' : optim_d.state_dict(),
                                     'steps' : steps,
                                     'epoch' : epoch})
                    
                # validation
                if steps % hp.validation_interval == 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking = True))
                            y_g_hat_mel = wav2melspec(y_g_hat.detach().cpu().numpy(), feature_mean, feature_std, hp)
                            y_g_hat_mel = torch.from_numpy(y_g_hat_mel.astype(float)).to(torch.float32).to(device, non_blocking = True)
                            y_g_hat_mel = y_g_hat_mel.squeeze(1)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
                            
                        val_err = val_err_tot / (j + 1)
                    
                    print('Validation Loss Total : {:4.3f}'.format(val_err))
                    
                    generator.train()
                    
            steps += 1
        
        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    
    hp = Hparams()
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        hp.num_gpus = torch.cuda.device_count()
        hp.batch_size = int(hp.batch_size / hp.num_gpus)
        print('Batch size per GPU : ', hp.batch_size)
        
    if hp.num_gpus > 1:
        mp.spawn(train, nprocs = hp.num_gpus, args = (hp,))
    else:
        train(0, hp)
        

if __name__ == '__main__':
    main()