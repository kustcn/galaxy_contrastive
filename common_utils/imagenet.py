import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as tf

class GalaxyData(datasets.ImageFolder):
    def __init__(self, root='/mnt/storage-ssd/liyadi/galaxy_data/',mode = 'train',
                transform=None):  # galaxy_decals_train  galaxy_SDSS
        super(GalaxyData, self).__init__(root=root+mode,
                                         transform=None)
        self.transform = transform 
        
        self.resize = tf.CenterCrop(256)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 
        'path':path,
        'meta': {'im_size': im_size, 'index': index}}
        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        #img = self.resize(img) 
        return img



