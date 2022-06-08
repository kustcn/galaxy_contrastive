import torch
import torch.nn as nn

class Contrastive_Loss(nn.Module):
    def __init__(self, temperature):
        super(Contrastive_Loss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda('cuda:1') # 生成对角线全为1 其他为0的矩阵
        unbind_features = torch.unbind(features, dim=1) # 按dim 维度切片
        contrast_features = torch.cat(unbind_features, dim=0) # 按dim 维度拼接
        anchor = features[:, 0] #原始图像的特征向量

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2) #在行维度重复1-1次，在列维度上重复2-1次
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda('cuda:1'), 0)
        mask = mask * logits_mask
        #scatter(input, dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中
        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask #前521列对角线元素(自己和自己的点积)设为0，
        
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # # Mean log-likelihood for positive 
        # loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
        # 同批次内 正负样本
        logits_positive = mask * logits # 所有正样本 下一步必须按行求和变为1列

        loss = - (logits_positive.sum(1, keepdim=True) - torch.log (exp_logits.sum(1, keepdim=True))).mean()
        
         
        return loss
