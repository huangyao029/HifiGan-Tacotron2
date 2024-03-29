import torch
import os
import glob


def init_weights(m, mean = 0., std = 0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)
        
        
        
def save_checkpoint(filepath, obj):
    print('Saving checkpoint to {}'.format(filepath))
    torch.save(obj, filepath)
    print('Complete.')
    
    
    
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print('Loading "{}"'.format(filepath))
    checkpoint_dict = torch.load(filepath, map_location = device)
    print('Complete.')
    return checkpoint_dict
        
        
        
def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
        
        
        
def get_padding(kernel_size, dilation = 1):
    return int((kernel_size * dilation - dilation) / 2)