import torch
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from .model import Encoder
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import pandas as pd
    
class SGCD():
    def __init__(self, 
        adata,
        device= torch.device('cpu'),
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=600, 
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        n=3,
        gama=0.9,
        similarity_metric='jsd'
        ):

        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.n = n
        self.gama = gama
        self.similarity_metric = similarity_metric
        
        fix_seed(self.random_seed)
        
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)
        
        if 'adj' not in adata.obsm.keys():
            construct_interaction(self.adata,self.n,self.gamma)
         
        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
    
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        self.adj = preprocess_adj(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)
        
    def train(self):
        self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        print('Begin to train ST data...')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
              
            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)
            
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)
            
            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)
            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
            self.model.eval()
            self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec

            return self.adata
