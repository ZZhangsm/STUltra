import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch_geometric.data import Data
import copy
import random

from .gat_conv import GATConv

cudnn.deterministic = True
cudnn.benchmark = True

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean) / std
        x = x * gamma + beta

        return x




def aug_feature_dropout(input_feat, drop_rate=0.2):
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_rate)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    if input_feat.is_sparse:
        coo = input_feat.coalesce()
        indices = coo.indices()
        values = coo.values()
        
        mask = ~torch.isin(indices[1], torch.tensor(drop_idx, device=indices.device))
        aug_input_feat = torch.sparse_coo_tensor(
            indices[:, mask], 
            values[mask], 
            size=input_feat.size(),
            device=input_feat.device
        )
    else:
        aug_input_feat[:, drop_idx] = 0
    return aug_input_feat

class Model(torch.nn.Module):
    def __init__(self, hidden_dims, beta=1, uncertainty=0.5):
        super(Model, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        
        self.perturbation = DistributionUncertainty(p=uncertainty) if uncertainty > 0 else nn.Identity()
                
        self.beta = beta
        self.discrimination_loss = nn.BCEWithLogitsLoss()
        self.DGI_projector = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, 20),
        )

 
    def forward(self, x, edge_index):
        h1 = F.elu(self.conv1(x, edge_index))
        h1 = self.perturbation(h1)  # perturbation
        h2 = self.conv2(h1, edge_index, attention=False)
        h2 = self.perturbation(h2)  # perturbation
        
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h3 = self.perturbation(h3)  # perturbation
        h4 = self.conv4(h3, edge_index, attention=False)
        h4 = self.perturbation(h4)  # perturbation

        return h2, h4

    def DGI(self, x, edge_index):
        x_aug = aug_feature_dropout(x, drop_rate=0.2)
        x_neg = self.encoding_mask_negative(x, keep_rate_negative=0)

        h1 = F.elu(self.conv1(x_aug, edge_index))
        h1 = self.conv2(h1, edge_index, attention=False)
        h1_neg = F.elu(self.conv1(x_neg, edge_index))
        h1_neg = self.conv2(h1_neg, edge_index, attention=False)
        
        h2 = self.DGI_projector(h1)
        h2_neg = self.DGI_projector(h1_neg)
        logit = torch.cat((h2.sum(1), h2_neg.sum(1)), 0)
        n = logit.shape[0] // 2
        disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0).to(logit.device)
        
        loss_disc = self.discrimination_loss(logit, disc_y)
        
        return loss_disc

    def encoding_mask_negative(self, x, keep_rate_negative=0):
        """Generate negative samples for both dense and sparse tensors"""
        num_nodes = x.size(0)
        device = x.device
        
        perm = torch.randperm(num_nodes, device=device)
        num_keep_nodes = int(keep_rate_negative * num_nodes)
        replace_nodes = perm[num_keep_nodes:] 
        num_replace = replace_nodes.shape[0]
        
        if num_replace == 0:
            return x 

        replace_nodes_idx = torch.randperm(num_nodes, device=device)[:num_replace]
        
        if x.is_sparse:
            coo = x.coalesce()
            indices = coo.indices()
            values = coo.values()
            new_indices = indices.clone()
            mapping = torch.arange(x.size(0), device=device)
            mapping[replace_nodes] = replace_nodes_idx
            
            new_indices[0] = mapping[indices[0]]
            
            out_x_negative = torch.sparse_coo_tensor(
                new_indices,
                values,
                size=x.size(),
                device=device
            )
        else:
            out_x_negative = x.clone()
            out_x_negative[replace_nodes] = x[replace_nodes_idx]
        return out_x_negative


    def train_step(self, x, edge_index, dgi_loss=True):
        h2, h4 = self.forward(x, edge_index)
        if x.is_sparse:
            recon_loss = F.mse_loss(x.to_dense(), h4)
        else:
            recon_loss = F.mse_loss(x, h4)
            
        if dgi_loss:
            dgi_loss = self.DGI(x, edge_index)
        else:
            dgi_loss = 0
        total_loss = recon_loss + self.beta * dgi_loss
        return total_loss, h2
    