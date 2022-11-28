import argparse,random
import os,time
import torch
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from common_utils.config import create_config
from common_utils.utils import build_model ,get_optimizer,adjust_learning_rate
from common_utils.memory import MemoryBank
from common_utils.train import train,fill_memory_bank
from common_utils.loss import Contrastive_Loss
from common_utils.imagenet import GalaxyData
from common_utils.utils import AugmentationDataset

config_file = './configs/experiment.yml'
# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_output',type=str,default='./output/',
                    help='Config file for the environment')
parser.add_argument('--config_exp',type=str,default=config_file,
                    help='Config file for the experiment')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    seed_everything(678)
    args = parser.parse_args()
    p = create_config(args.config_output, args.config_exp)
    # Model
    model = build_model(p)
    print(model)
    
    model.cuda('cuda:1')
   
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    train_transforms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.RandomCrop(180),
                transforms.Resize(84),
                transforms.RandomResizedCrop(size=84,scale=(0.8,0.8)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation']['color_jitter'])
                ], p=p['augmentation']['random_apply']['p']),
                transforms.RandomGrayscale(**p['augmentation']['random_grayscale']),
                transforms.RandomRotation(180), 
                transforms.ToTensor(),
                transforms.Normalize(**p['augmentation']['normalize'])
        ])
    
    val_transforms = transforms.Compose([
                transforms.CenterCrop(180),
                transforms.Resize(84),
                transforms.ToTensor(), 
                transforms.Normalize(**p['augmentation']['normalize'])
            ])
    
    train_dataset = GalaxyData(mode = 'train',transform = train_transforms) 
    train_dataset = AugmentationDataset(train_dataset)
    val_dataset = GalaxyData(mode = 'val',transform = val_transforms) 
    test_dataset = GalaxyData(mode = 'test',transform = val_transforms)

    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, 
                    num_workers=p['num_workers'],
                    batch_size=p['batch_size'], pin_memory=True, 
                    drop_last=True, shuffle=True)
    
    val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    num_workers=p['num_workers'],
                    batch_size=p['batch_size'], pin_memory=True, 
                    drop_last=False, shuffle=False)
    
    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    num_workers=p['num_workers'],
                    batch_size=p['batch_size'], pin_memory=True, 
                    drop_last=False, shuffle=False)
    
    print('Dataset contains {}/{}/{} train/val/test samples'.format(len(train_dataset), len(val_dataset),len(test_dataset)))
    

    # Memory Bank
    print('Build MemoryBank')
    base_dataset = GalaxyData(mode = 'train',transform = val_transforms)

    base_dataloader = torch.utils.data.DataLoader(
                    base_dataset,
                    num_workers=p['num_workers'],
                    batch_size=p['batch_size'], pin_memory=True, 
                    drop_last=False, shuffle=False)
     
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['features_dim'],
                                p['num_classes'], 
                                p['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['features_dim'],
                                p['num_classes'],
                                 p['temperature'])
    memory_bank_val.cuda()

    memory_bank_test = MemoryBank(len(test_dataset),
                                p['features_dim'],
                                p['num_classes'], 
                                p['temperature'])
    memory_bank_test.cuda()


    # Criterion

    criterion = Contrastive_Loss(p['temperature'])
    criterion = criterion.cuda('cuda:1')

    # Optimizer and scheduler
    optimizer = get_optimizer(p, model)
    
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print('Restart from checkpoint {}'.format(p['pretext_checkpoint']))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda('cuda:1')
        acc_train = checkpoint['acc_train']
        acc_val = checkpoint['acc_val']
        nmi_train = checkpoint['nmi_train']
        ari_train = checkpoint['ari_train']
        nmi_val = checkpoint['nmi_val']
        ari_val = checkpoint['ari_val']
        loss_list = checkpoint['loss_list']
        start_epoch = checkpoint['epoch']

    else:
        print('No checkpoint file at {}'.format(p['pretext_checkpoint']))
        start_epoch = 0
        acc_train = []
        acc_val = []
        loss_list = []
        nmi_train = []
        ari_train = []
        nmi_val = []
        ari_val = []
        
    
    # Training
    print('Main loop starts-------')
    
    for epoch in range(start_epoch, p['epochs']):
        st = time.time()
        print('Epoch %d/%d' %(epoch, p['epochs']))
        

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Training ...')
        
        loss = train(train_dataloader, model, criterion, optimizer, epoch)

        # Fill memory bank
        print('Fill memory bank for kNN...')

        # train_set
        fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate 
        print('Evaluate ...')
        topk = 15
        print('Mine the nearest neighbors (Top-%d) on train_set' %(topk)) 
        model.eval()
        _,acc1,nmi,ari,_ = memory_bank_base.mine_nearest_neighbors(topk)
        
        acc_train.append(acc1)
        nmi_train.append(nmi)
        ari_train.append(ari)
        print('     ACC:%.5f NMI:%.5f ARI:%.5f' %(acc1,nmi,ari) )

        # val_set
        fill_memory_bank(val_dataloader, model, memory_bank_val)
        
        print('Mine the nearest neighbors (Top-%d) on val_set' %(topk)) 

        _,acc2,nmi,ari,_ = memory_bank_val.mine_nearest_neighbors(topk)
        acc_val.append(acc2)
        nmi_val.append(nmi)
        ari_val.append(ari)
        print('     ACC:%.5f  NMI:%.5f  ARI:%.5f' %(acc2,nmi,ari) )
        # Checkpoint
        loss_list.append(loss)
        print('time cost:',time.time()-st)
        if (epoch+1) % 5 == 0:
         print('Checkpoint ...')
         torch.save({'optimizer': optimizer.state_dict(), 
                    'model': model.state_dict(),
                    'acc_train' :acc_train,
                    'acc_val':acc_val,
                    'nmi_train' : nmi_train,
                    'ari_train' : ari_train,
                    'nmi_val' :nmi_val,
                    'ari_val' :ari_val,
                    'loss_list':loss_list,
                    'epoch': epoch + 1},
                     p['pretext_checkpoint'])


   
    #test_set
    print('Fill memory bank for test dataset ...')   

    fill_memory_bank(test_dataloader, model, memory_bank_test)

    topk = 15    

    print('Mine the nearest neighbors (Top-%d)' %(topk)) 

    indices,acc3,nmi,ari,res= memory_bank_test.mine_nearest_neighbors(topk)

    print('Result of test_set is ACC:%.5f  NMI:%.5f  ARI:%.5f' %(acc3,nmi,ari) )

    np.save(p['topk_neighbors_test'], indices)

    np.save(p['pre_target'],res)
 

 
if __name__ == '__main__':
    main()
