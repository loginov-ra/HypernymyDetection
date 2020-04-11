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


def make_model_vocab(embedder=None, thesaurus=None, cut_most_common=27):
    if embedder is None:
        embedder = ft.load_model('../data/models/fasttext_deeppavlov.bin')    
    if thesaurus is None:
        thesaurus = ThesaurusParser("../data/RuThes", need_closure=False)
        
    vocab_embeddings = dict()
    for _, entry_dict in thesaurus.text_entries.items():
        lemma = entry_dict['lemma']
        vocab_embeddings[lemma] = embedder.get_sentence_vector(lemma)
    
    DIR_PATH = "/home/loginov-ra/MIPT/HypernymyDetection/data/Lenta/texts_tagged_processed_tree"
    file_list = os.listdir(DIR_PATH)
    file_list = [join(DIR_PATH, filename) for filename in file_list]
    
    word_ctr = Counter()
    no_deeppavlov = 0

    for filename in tqdm(file_list):
        with open(filename, encoding='utf-8') as sentences_file:
            sentences = json.load(sentences_file)
            for sent in sentences:
                if 'deeppavlov' not in sent:
                    no_deeppavlov += 1
                    continue

                multitokens, _ = sent['multi']
                for t in multitokens:
                    word_ctr[t] += 1
    
    additional_words = word_ctr.most_common(n=100000)[cut_most_common:]
    for word, _ in additional_words:
        vocab_embeddings[word] = embedder.get_word_vector(word)
        
    return vocab_embeddings


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
        probas = torch.sigmoid(logits)
        return probas