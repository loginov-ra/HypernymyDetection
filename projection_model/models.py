import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import fasttext as ft
from thesaurus_parsing.thesaurus_parser import ThesaurusParser
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