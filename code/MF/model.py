from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pytorch_lightning import LightningModule


def get_model(params, return_instance=True):
    """Acts as lookup table of all the models that are implemented in 
    the module."""

    if params["alias"] == "MF":
        return (MF(
            params["n_users"], params["n_items"],
            params["n_latents"], 
            params["batch_size"], params["n_epochs_offline"], 
            params["n_epochs_online"], params["early_stopping_online"],
            learning_rate=params["learning_rate"], 
            l2_regularization_constant=params["l2_regularization_constant"])
            if return_instance else MF)
    elif params["alias"] == "SP":
        return (SingleParam() if return_instance else SingleParam)
    elif params["alias"] == "UP":
        return (UserParam(params["n_users"]) if return_instance else UserParam)
    elif params["alias"] == "IP":
        return (ItemParam(params["n_items"]) if return_instance else ItemParam)


class MF(LightningModule):
    def __init__(
        self, n_users, n_items, n_latent, 
        batch_size, n_epochs_offline, n_epochs_online, 
        early_stopping_online,
        learning_rate=1e-3,
        l2_regularization_constant=1e-1):

        # initialize hyperparameters
        self.learning_rate = learning_rate
        self.l2_regularization_constant = l2_regularization_constant

        # initialize variables
        super(MF, self).__init__()
        self.user_bias_emb = nn.Embedding(n_users, 1)
        self.item_bias_emb = nn.Embedding(n_items, 1)
        self.user_latent_emb = nn.Embedding(n_users, n_latent)
        self.item_latent_emb = nn.Embedding(n_items, n_latent)

        # set bias initialization to zero
        nn.init.zeros_(self.user_bias_emb.weight)
        nn.init.zeros_(self.item_bias_emb.weight)

        # register instance params
        self.n_users = n_users
        self.n_items = n_items
        self.n_latent = n_latent

        # required by loading from checkpoint (LightningModule)
        self.save_hyperparameters() 

    def forward(self, x):
        # parse input
        user_ids = x[:, 0].squeeze()
        item_ids = x[:, 1].squeeze()

        # compute logits
        user_bias = self.user_bias_emb(user_ids).squeeze()
        item_bias = self.item_bias_emb(item_ids).squeeze()
        user_latent = self.user_latent_emb(user_ids)
        item_latent = self.item_latent_emb(item_ids)
        interaction = torch.mul(user_latent, item_latent).sum(-1)
        # replace for nn.Sigmoid
        return torch.sigmoid(user_bias + item_bias + interaction)

    def reset_parameters(self):
        # by default initialized with standard Gaussian
        self.user_latent_emb.reset_parameters()
        self.item_latent_emb.reset_parameters()

        # set bias initialization to zero
        nn.init.zeros_(self.user_bias_emb.weight)
        nn.init.zeros_(self.item_bias_emb.weight)

    def training_step(self, batch, batch_idx):

        # parse input
        x = batch[0][:, 0:2]
        y = batch[1]

        # get loss
        scores = self(x)
        ce_loss = F.binary_cross_entropy(scores, y)
        # l2_loss =  torch.norm(self.user_latent_emb) + torch.norm(
        #     self.item_latent_emb) + torch.sum(self.user_bias_emb ** 2) + \
        #         torch.sum(self.item_bias_emb ** 2)
        loss = ce_loss  # + self.l2_regularization_constant * l2_loss

        self.log("train_loss", ce_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # parse input
        x = batch[0][:, 0:2]
        y = batch[1]

        # get loss
        scores = self(x)
        loss = F.binary_cross_entropy(scores, y)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):

        # parse input
        x = batch[0][:, 0:2]
        y = batch[1]

        # get loss
        scores = self(x)
        test_loss = F.binary_cross_entropy(scores, y)

        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(), lr=self.learning_rate,
            weight_decay=self.l2_regularization_constant)


class SingleParam(nn.Module):
    def __init__(self):
        super(SingleParam, self).__init__()
        self.param = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, user_ids, item_ids):
        return torch.sigmoid(self.param).expand(user_ids.size())

    def reset_parameters(self):
        self.param = nn.Parameter(torch.zeros(1))


class UserParam(nn.Module):
    def __init__(self, n_users):
        super(UserParam, self).__init__()
        self.user_emb = nn.Embedding(n_users, 1)

        # initialize weights at zero
        nn.init.zeros_(self.user_emb.weight)

    def forward(self, user_ids, item_ids):
        return torch.sigmoid(self.user_emb(user_ids).squeeze())

    def reset_parameters(self):
        nn.init.zeros_(self.user_emb.weight)


class ItemParam(nn.Module):
    def __init__(self, n_items):
        super(ItemParam, self).__init__()
        self.item_emb = nn.Embedding(n_items, 1)

        # initialize weights at zero
        nn.init.zeros_(self.item_emb.weight)

    def forward(self, user_ids, item_ids):
        return torch.sigmoid(self.item_emb(item_ids).squeeze())

    def reset_parameters(self):
        nn.init.zeros_(self.item_emb.weight)
