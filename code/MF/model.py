import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np


class MF(nn.Module):
    def __init__(self, n_user, n_item, n_latent):
        
        #initialize variables
        super().__init__()
        self.user_bias_emb = nn.Embedding(n_user, 1)
        self.item_bias_emb = nn.Embedding(n_item, 1)
        self.user_latent_emb = nn.Embedding(n_user, n_latent)
        self.item_latent_emb = nn.Embedding(n_item, n_latent)

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
        return F.sigmoid(user_bias + item_bias + interaction)
    

