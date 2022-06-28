import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np


class MF(nn.Module):
    def __init__(self, n_user, n_item, n_latent):
        
        #initialize variables
        super(MF, self).__init__()
        self.user_bias_emb = nn.Embedding(n_user, 1)
        self.item_bias_emb = nn.Embedding(n_item, 1)
        self.user_latent_emb = nn.Embedding(n_user, n_latent)
        self.item_latent_emb = nn.Embedding(n_item, n_latent)

        ## set bias initialization to zero
        nn.init.zeros_(self.user_bias_emb.weight)
        nn.init.zeros_(self.item_bias_emb.weight)
        
        # register instance params
        self.n_user = n_user
        self.n_item = n_item
        self.n_latent = n_latent

    def forward(self, user_ids, item_ids):
        user_bias = self.user_bias_emb(user_ids).squeeze()
        item_bias = self.item_bias_emb(item_ids).squeeze()
        user_latent = self.user_latent_emb(user_ids)
        item_latent = self.item_latent_emb(item_ids)
        interaction = torch.mul(user_latent, item_latent).sum(-1)
        return torch.sigmoid(user_bias + item_bias + interaction) # replace for nn.Sigmoid

    def reset_parameters(self):
        # by default initialized with standard Gaussian
        self.user_latent_emb.reset_parameters()
        self.item_latent_emb.reset_parameters()
        
        ## set bias initialization to zero
        nn.init.zeros_(self.user_bias_emb.weight)
        nn.init.zeros_(self.item_bias_emb.weight)
    

