import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import fasttext as ft
from vocab import Vocab
import os
import json
from os.path import join
from collections import Counter
from tqdm import tqdm_notebook as tqdm


class CRIMModel(nn.Module):
    def __init__(self, n_matrices=5, embedding_dim=300, init_sigma=0.01):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_matrices = n_matrices
        self.init_sigma = init_sigma
        
        matrix_shape = (n_matrices, 1, embedding_dim, embedding_dim)
        self.matrices = torch.FloatTensor(size=matrix_shape)
        self.prob_layer = nn.Linear(in_features=n_matrices, out_features=1)
        
        for i in range(n_matrices):
            eye_tensor = torch.FloatTensor(size=(embedding_dim, embedding_dim))
            noise_tensor = torch.FloatTensor(size=(embedding_dim, embedding_dim))
            torch.nn.init.eye_(eye_tensor)
            torch.nn.init.normal_(noise_tensor, std=init_sigma)
            self.matrices[i][0] = eye_tensor + noise_tensor
            
        torch.nn.init.normal_(self.prob_layer.weight, std=0.1)
        self.matrices = nn.Parameter(self.matrices.requires_grad_())
        
    def forward(self, input_dict):
        candidate = input_dict['candidate']
        candidate_batch = candidate.shape[0]
        candidate = candidate.view((candidate_batch, 1, self.embedding_dim))
        batch = input_dict['batch'].unsqueeze(-1)
        batch_size = batch.shape[0]
        projections = torch.matmul(self.matrices, batch).permute(1, 0, 2, 3).squeeze(-1)
        similarities = F.cosine_similarity(projections, candidate, dim=-1)
        logits = self.prob_layer(similarities)
        return logits
    

class CRIMModelVocab(nn.Module):
    def __init__(self, vocab, n_matrices=5, embedding_dim=300, init_sigma=0.01):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_matrices = n_matrices
        self.init_sigma = init_sigma
        
        matrix_shape = (n_matrices, 1, embedding_dim, embedding_dim)
        self.matrices = torch.FloatTensor(size=matrix_shape)
        self.prob_layer = nn.Linear(in_features=n_matrices, out_features=1)
        
        for i in range(n_matrices):
            eye_tensor = torch.FloatTensor(size=(embedding_dim, embedding_dim))
            noise_tensor = torch.FloatTensor(size=(embedding_dim, embedding_dim))
            torch.nn.init.eye_(eye_tensor)
            torch.nn.init.normal_(noise_tensor, std=init_sigma)
            self.matrices[i][0] = eye_tensor + noise_tensor
            
        torch.nn.init.normal_(self.prob_layer.weight, std=0.1)
        self.matrices = nn.Parameter(self.matrices.requires_grad_())
        
        emb_matrix, pad_idx = vocab.get_embedding_table()
        emb_tensor = torch.FloatTensor(emb_matrix)
        self.embedding_layer = nn.Embedding.from_pretrained(emb_tensor, freeze=True, padding_idx=pad_idx)
        self.vocab = vocab
        self.pad_idx = pad_idx
    
    def batch_to_embeddings(self, indices_batch):
        embeddings = self.embedding_layer(indices_batch)
        not_pad = indices_batch != self.pad_idx
        emb_sum = torch.sum(embeddings, dim=1)
        not_pad_sum = torch.sum(not_pad, dim=1).unsqueeze(1)
        not_pad_sum[not_pad_sum == 0] = 1
        return emb_sum / not_pad_sum.float()
    
    def forward(self, input_dict):
        candidate = input_dict['candidate']
        candidate = self.batch_to_embeddings(candidate)
        candidate_batch = candidate.shape[0]
        candidate = candidate.view((candidate_batch, 1, self.embedding_dim))
        
        batch = input_dict['batch']
        batch = self.batch_to_embeddings(batch).unsqueeze(-1)
        batch_size = batch.shape[0]
        
        projections = torch.matmul(self.matrices, batch).permute(1, 0, 2, 3).squeeze(-1)
        similarities = F.cosine_similarity(projections, candidate, dim=-1)
        logits = self.prob_layer(similarities)
        return logits