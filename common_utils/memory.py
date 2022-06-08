import numpy as np
import torch
from sklearn import metrics

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.path = []
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    
    @torch.no_grad()
    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        _, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) 
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) 
            # anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            # accuracy = np.mean(neighbor_targets == anchor_targets)
            pre_label = []
            accuracy = 0
            for item in neighbor_targets:
                counts = np.bincount(item)
                for m,n in enumerate(counts):
                    if n == max(counts):
                        pre_label.append(m)
                        break

            pre_label = np.array(pre_label)          
            for i in range(len(targets)):
                if pre_label[i] == targets[i]:
                    accuracy+=1
            nmi = metrics.normalized_mutual_info_score(targets, pre_label)
            ari = metrics.adjusted_rand_score(targets, pre_label)
    
            return indices, 100*accuracy/len(targets),nmi,ari,[pre_label,np.array(targets),self.path]
       
        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets,path):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.path.extend(path)
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:1')
