import sys
sys.path.append('../')

from collections import Counter
import fasttext as ft
import json
import os
from thesaurus_parsing.thesaurus_parser import ThesaurusParser
from tqdm import tqdm_notebook as tqdm
import numpy as np


class Vocab:
    def __init__(self, thesaurus=None, embedder=None, empty=False, n_texts=None, add_size=100000, cnt_most_common=27):
        self.word_to_id = dict()
        self.id_to_word = []
        self.embeddings = dict()
        
        self.n_texts = n_texts
        self.add_size = add_size
        self.cnt_most_common = cnt_most_common
        
        if not empty:
            if thesaurus is None:
                thesaurus = ThesaurusParser("../data/RuThes", need_closure=False)
            self.embedder = embedder
            self.thesaurus = thesaurus
            self.popular_multitokens = self.get_popular_tokens()
            self.build_word_dictionary()
            self.hardcode_padding()
    
    def get_popular_tokens(self):
        popular_multitokens = set()
        for _, entry_dict in self.thesaurus.text_entries.items():
            lemma = entry_dict['lemma']
            popular_multitokens.add(lemma)

        DIR_PATH = "/home/loginov-ra/MIPT/HypernymyDetection/data/Lenta/texts_tagged_processed_tree"
        file_list = os.listdir(DIR_PATH)
        file_list = [os.path.join(DIR_PATH, filename) for filename in file_list]
        word_ctr = Counter()
        
        if self.n_texts is None:
            self.n_tests = len(file_list)
        
        for filename in tqdm(file_list[:self.n_texts]):
            with open(filename, encoding='utf-8') as sentences_file:
                sentences = json.load(sentences_file)
                for sent in sentences:
                    if 'deeppavlov' not in sent:
                        continue

                    multitokens, _ = sent['multi']
                    for t in multitokens:
                        word_ctr[t] += 1

        additional_words = word_ctr.most_common(n=self.add_size)[self.cnt_most_common:]
        for word, _ in additional_words:
            popular_multitokens.add(word)
            
        return popular_multitokens
    
    def hardcode_padding(self):
        emb_dim = self.embeddings[self.id_to_word[0]].size
        self.embeddings['[PAD]'] = np.zeros(emb_dim)
    
    def add_token_to_dict(self, word, embed=None):
        if word not in self.word_to_id:
            self.word_to_id[word] = len(self.id_to_word)
            self.id_to_word.append(word)
            if embed is None:
                self.embeddings[word] = self.embedder(word)
            else:
                self.embeddings[word] = embed
    
    def build_word_dictionary(self):
        self.add_token_to_dict('[UNK]')
        self.add_token_to_dict('[PAD]')
        for multitoken in self.popular_multitokens:
            for token in multitoken.split():
                self.add_token_to_dict(token)
    
    def serialize_to_file(self, file):
        json_object = []
        for i, word in enumerate(self.id_to_word):
            json_object.append({
                'id': i,
                'token': word,
                'embedding': self.embeddings[word].tolist()
            })
        json.dump(json_object, file, ensure_ascii=False)
        
    def construct_from_file(self, file):
        self.word_to_id = dict()
        self.id_to_word = []
        self.embeddings = dict()
        json_object = json.load(file)
        for i, item in enumerate(json_object):
            word = item['token']
            embed = item['embedding']
            self.add_token_to_dict(word, embed)
            
    def process_batch(self, batch):
        """Preprocess multitoken batch
        
        Get indices for tokens inside batch
        All the sequences are padded with the [PAD] token
        
        Parameters
        ----------
        batch : Iterable[str]
            batch of multitokens to pass through the network
            
        Returns
        -------
        2-d array: indices for words in multitokens
        """
        res = []
        max_len = 0
        
        for multitoken in batch:
            token_indices = []
            for token in multitoken.split():
                if token in self.word_to_id:
                    token_indices.append(self.word_to_id[token])
                else:
                    token_indices.append(self.word_to_id['[UNK]'])
            res.append(token_indices)
            max_len = max(max_len, len(token_indices))
        
        for i in range(len(res)):
            for _ in range(max_len - len(res[i])):
                res[i].append(self.word_to_id['[PAD]'])
            
        return res
    
    def get_embedding_table(self):
        """Get table of embeds for nn.Embedding layer
        
        Returns
        -------
        weigths : 2d-array
            Matrix, each row represents the embedding
            The order used is fixed inside the vocabulary
        pad_idx : int
            Index of padding
        """
        embedding_matrix = []
        for word in self.id_to_word:
            embedding_matrix.append(self.embeddings[word])
        return np.array(embedding_matrix), self.word_to_id['[PAD]']