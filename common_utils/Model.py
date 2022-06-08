import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from linformer import Linformer
from efficient import ViT

class GalaxyModel(nn.Module):
    def __init__(self, backbone, features_dim=128):
        super(GalaxyModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        
 
        
        self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.BatchNorm1d(self.backbone_dim),
                    nn.ReLU(),
                    nn.Linear(self.backbone_dim, features_dim))
        
        

    def forward(self, x):
        x=self.backbone(x)
        features = self.contrastive_head(x)
        features = F.normalize(features, dim = 1)
        return features


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.adaptmaxpool = nn.AdaptiveMaxPool2d((1,1))


        self.eff_transformer  = Linformer(
                    dim=128,
                    seq_len=49+1,  # 7x7 patches + 1 cls-token
                    depth=6,
                    heads=4,
                    k=64
                                    )
        self.vit = ViT(
                    dim=128,
                    image_size=84,
                    patch_size=12,
                    num_classes=3,
                    transformer= self.eff_transformer,
                    channels=3,
                        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x_vit = self.vit(x)
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        # out = self.maxpool(out)
        # out = self.layer1(out)        
        out = self.layer2(out)      
        out = self.layer3(out)      
        #out = self.layer4(out)    
        out = self.avgpool(out)
        #out = self.adaptmaxpool(out)
        out = torch.flatten(out, 1)
        out = torch.cat((out,x_vit),dim =1)
        return out
        # return x_vit

def resnet18():
    return {'backbone': ResNet(BasicBlock, [2, 2, 2, 2]), 'dim': 512}
