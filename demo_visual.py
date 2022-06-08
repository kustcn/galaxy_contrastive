from visualizer import get_local
get_local.activate()

import torch
import torchvision.transforms as T
#from timm.models.vision_transformer import vit_small_patch16_224
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import argparse,random
import os,time
#from sqlite3 import Time
#from time import time
import torch
import numpy as np
import torch.distributed as dist
from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate,get_test_dataset
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

rootpath = '/mnt/storage-ssd/liyadi/Unsupervised-res_vit/configs/pretext/simclr_cifar10.yml'
# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',type=str,default='/mnt/storage-ssd/liyadi/Unsupervised-res_vit/configs/env.yml',
                    help='Config file for the environment')
parser.add_argument('--config_exp',type=str,default=rootpath,
                    help='Config file for the experiment')

#parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(678)
args = parser.parse_args()
p = create_config(args.config_env, args.config_exp)
print(colored(p, 'red'))


# Model
print(colored('Retrieve model', 'blue'))
model = get_model(p)
print('Model is {}'.format(model.__class__.__name__))
print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
print(model)
# Criterion
print(colored('Retrieve criterion', 'blue'))
criterion = get_criterion(p)
print('Criterion is {}'.format(criterion.__class__.__name__))


# Optimizer and scheduler
print(colored('Retrieve optimizer', 'blue'))
optimizer = get_optimizer(p, model)
print(optimizer)
# Checkpoint
if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        #optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        
        acc_train = checkpoint['acc_train']
        acc_val = checkpoint['acc_val']
        nmi_train = checkpoint['nmi_train']
        ari_train = checkpoint['ari_train']
        nmi_val = checkpoint['nmi_val']
        ari_val = checkpoint['ari_val']
        loss_list = checkpoint['loss_list']
        start_epoch = checkpoint['epoch']

else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        acc_train = []
        acc_val = []
        loss_list = []
        nmi_train = []
        ari_train = []
        nmi_val = []
        ari_val = []
        

def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask
    

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=12, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    size = image.size
    for i in range(12):
        for j in range(8):

            attention_map = att_map[i][0,j,:,:][105]
            cls_weight = attention_map[0]
    
            mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
            mask = mask / max(np.max(mask),cls_weight)
            mask = Image.fromarray(mask).resize((size))
            st= time.time()
            image = np.array(image)
    # image_1 = image[:,:,0]
    # image_1 = Image.fromarray(image_1)
            mask = (np.array(mask))
    # mask = (mask-mask.min())*255/(mask.max()-mask.min())
    
            padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)

    # if grid_index != 0: # adjust grid_index since we pad our image
    #     grid_index = grid_index + (grid_index-1) // grid_size[1]
        
            grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    # fig, ax = plt.subplots(1, 2, figsize=(10,7))
    # fig.tight_layout()
    
    # ax[0].imshow(grid_image)
    # ax[0].axis('off')
    
    # ax[1].imshow(mask)
    # # ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    # ax[1].imshow(image)
    # ax[1].axis('off')
    # plt.imsave('./11111111111111111.png',image_1)
    # plt.imsave('./padded_mask.png',padded_mask)
    # plt.imsave('./meta_mask.png',meta_mask)
    
    # ax[0].imshow(grid_image)
    # ax[0].axis('off')
            plt.axis('off')
            plt.imshow(image,cmap ='bone')
            plt.imshow(mask, alpha=alpha, cmap='rainbow')
    # plt.imsave('./1111111111111111111.png',np.array(grid_image))
    # plt.imsave('./33333333333.png',np.array(mask))
            plt.savefig('./U/'+str(time.time()-st)+'.jpg')
    # ax[1].imshow(meta_mask)
    
    

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    with_cls_token = False
      
    grid_image = highlight_grid(image, [grid_index], grid_size)
    
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()
    # plt.imsave('./grid_image2.png',grid_image)
    # plt.imsave('./mask-np.max(mask).png',mask/np.max(mask))
    
    
def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1])) 
        #已知原数组坐标，现在为矩阵时的坐标值
        a= ImageDraw.ImageDraw(image)
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image


image = Image.open('/mnt/storage-ssd/liyadi/Unsupervised-res_vit/84.jpg').convert('RGB')

#imagenet_cls = json.load(open('/mnt/storage-ssd/liyadi/Visualizer-main/imagenet_cls.json'))

normalize = T.Normalize(mean=[0.0430, 0.0379, 0.0286], #[0.485, 0.456, 0.406],
                        std=[0.0873, 0.0724, 0.0648]   #[0.229, 0.224, 0.225]
                        )
transforms = T.Compose([
            
            # T.CenterCrop(210),
            # T.Resize(128),
            T.ToTensor(),
            normalize,
        ])

input_tensor = transforms(image).unsqueeze(0)

get_local.clear()
with torch.no_grad():
    #vit = vit_small_patch16_224(pretrained=True)
    vit = model.backbone.vit
    #print(vit)
    out = vit(input_tensor)
    
# print('Top1 prediction:')
# print(imagenet_cls[str(out.argmax().item())])

cache = get_local.cache
print(list(cache.keys()))

attention_maps = cache['LinformerSelfAttention.forward']

visualize_grid_to_grid_with_cls(attention_maps,grid_index=0,image=image)

