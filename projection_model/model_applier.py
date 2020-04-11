from collections import defaultdict
from abc import ABC, abstractmethod
from tqdm import tqdm_notebook as tqdm
import numpy as np
import json


class ModelApplier(ABC):
    def _fill_vocab_indices(self):
        self.idx_to_word = []
        self.word_to_idx = dict()
        
        for word in self.vocab.keys():
            self.word_to_idx[word] = len(self.idx_to_word)
            self.idx_to_word.append(word)
    
    def __init__(self, vocab_dict):
        self.vocab = vocab_dict
        self._fill_vocab_indices()
    
    @abstractmethod
    def apply_model_to_query(self, query):
        """Apply model to query
        
        Get ranked list of candidate indices from vocabulary
        
        Parameters
        ----------
        query : str
            the query for model application
            
        Returns
        -------
        List of vocab indices of ranked candidates
        """
        pass
    
    def _leave_top_apply_to_query(self, query, leave_top):
        return self.apply_model_to_query(query)[:leave_top]
    
    def load_correct_answers(self, correct_answers):
        """Fill correct answers for comfortable application
        
        Makes dictionary of sets with correct answers
        
        Parameters
        ----------
        correct_answers : dict[str, list[str]]
            correct answers from hypernymy dictionary
            
        Returns
        -------
        no value
        """
        self.correct_answers = defaultdict(set)
        for hyponym, hypernyms in correct_answers.items():
            for hypernym in hypernyms:
                hypernym_idx = self.word_to_idx[hypernym]
                self.correct_answers[hyponym].add(hypernym_idx)
                
    def apply(self, test_queries, leave_top=1000):
        """Apply model to a set of queries
        
        Applies model to test set and saves the results inside
        
        Parameters
        ----------
        test_queries : list[str]
            queries for test, queries should have correct answers loaded
            
        leave_top: int
            leave only this number of top ranked, because others are useless
            
        Returns
        -------
        no value
        """
        self.test_queries = test_queries
        self.results = np.zeros((len(test_queries), leave_top))
        for i, q in enumerate(tqdm(test_queries)):
            answer = self._leave_top_apply_to_query(q, leave_top)
            self.results[i] = answer
            del answer
        
    def get_accuracy(self, k=None):
        """Calculate accuracy
        
        Calculates accuracy at k (Acc@K) metric
        
        Parameters
        ----------
        k : optional[int]
            number of first ranked candidates to include. If None, defaults to 1
        
        Returns
        -------
        Accuracy value for calculated result, float
        """
        k = 1 if k is None else k
        n_correct = 0
        
        for query, answer in tqdm(zip(self.test_queries, self.results)):
            correct_set = self.correct_answers[query]
            is_correct = False
            for candidate in answer[:k]:
                if candidate in correct_set:
                    is_correct = True
                    break
            n_correct += int(is_correct)
            
        return n_correct / len(self.test_queries)
            
    def get_precision(self, k=None):
        """Calculate precision
        
        Calculates precison at k (P@K) metric
        
        Parameters
        ----------
        k : optional[int]
            number of first ranked candidates to include. If None, defaults to 1
        
        Returns
        -------
        Precision value for calculated result, float
        """
        k = 1 if k is None else k
        precisions = []
        
        for query, answer in tqdm(zip(self.test_queries, self.results)):
            correct_set = self.correct_answers[query]
            n_relevant = 0
            for candidate in answer[:k]:
                if candidate in correct_set:
                    n_relevant += 1
            precisions.append(n_relevant / k)
            
        return np.mean(precisions)
    
    def get_recall(self, k=None):
        """Calculate recall
        
        Calculates recall at k (R@K) metric
        
        Parameters
        ----------
        k : optional[int]
            number of first ranked candidates to include. If None, defaults to 1
        
        Returns
        -------
        Recall value for calculated result, float
        """
        k = 1 if k is None else k
        recalls = []
        
        for query, answer in tqdm(zip(self.test_queries, self.results)):
            correct_set = self.correct_answers[query]
            n_relevant = 0
            for candidate in answer[:k]:
                if candidate in correct_set:
                    n_relevant += 1
            recalls.append(n_relevant / len(correct_set))
            
        return np.mean(recalls)
    
    def get_map(self):
        """Calculate MAP
        
        Calculates mean average precision metric
        Formulas: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#f9ce
        
        Returns
        -------
        MAP value for calculated result, float
        """
        average_precisions = []
        
        for query, answer in tqdm(zip(self.test_queries, self.results)):
            correct_set = self.correct_answers[query]
            average_precision = 0
            n_relevant = 0
            for i, candidate in enumerate(answer):
                if candidate in correct_set:
                    n_relevant += 1
                    average_precision += (n_relevant / (i + 1))
            average_precision /= len(correct_set)
            average_precisions.append(average_precision)
            
        return np.mean(average_precisions)