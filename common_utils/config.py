
import os
import yaml
from easydict import EasyDict

def create_config(output, config_file_exp):
       
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()   
    # Copy
    for k, v in config.items():
        cfg[k] = v
    
    cfg['pretext_checkpoint'] = os.path.join(output, 'checkpoint.pth.tar')
    cfg['topk_neighbors_test'] = os.path.join(output, 'topk-test-neighbors.npy')
    cfg['pre_target'] = os.path.join(output, 'pre_target.npy')
    
    return cfg 
