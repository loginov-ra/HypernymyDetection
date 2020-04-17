import sys
sys.path.append("../")

import numpy as np
from deeppavlov import build_model, configs
from nltk import WordPunctTokenizer, MWETokenizer
from os import listdir
from os.path import join
from tqdm import tqdm_notebook as tqdm
import json
from thesaurus_parsing.thesaurus_parser import ThesaurusParser
from syntax_trees.syntax_tree import SyntaxTree
import os


class ParsedSentence:
    def __init__(self, init_tokens, tagged_lemma, thesaurus, deeppavlov_lemma=None,
                 deeppavlov_pos=None, syntax_model=None):
        self.init_tokens = init_tokens
        self.tagged_lemma = tagged_lemma
        self.deeppavlov_lemma = deeppavlov_lemma
        self.deeppavlov_pos = deeppavlov_pos
        self.syntax_tree = SyntaxTree(syntax_model) if syntax_model else None
        self.thesaurus = thesaurus
        self.mwe_tokenizer = thesaurus.mwe_tokenizer
        
        self.make_multitokens()
        if self.syntax_tree:
            self.parse_syntax()
                
    def parse_syntax(self):
        self.syntax_tree.load(self.init_tokens)
        try:
            self.syntax_tree.build()
        except:
            self.syntax_tree = None
    
    def make_multitokens(self):
        tokens = self.tagged_lemma if self.deeppavlov_lemma is None else self.deeppavlov_lemma
        multitokens = self.mwe_tokenizer.tokenize(tokens)
        
        multi_to_main_token = [0] * len(multitokens)
        start_token_id = 0
        
        for i, multitoken in enumerate(multitokens):
            parts = multitoken.split()
            
            if multitoken not in self.thesaurus.lemma_to_entry or len(parts) == 1:
                multi_to_main_token[i] = start_token_id
            else:
                entry_id = self.thesaurus.lemma_to_entry[multitoken]
                main_word = self.thesaurus.text_entries[entry_id]["main_word"]
                try:
                    part_id = parts.index(main_word)
                except:
                    part_id = 0
                multi_to_main_token[i] = start_token_id + part_id
                
            start_token_id += len(parts)
            
        self.multitokens = multitokens
        self.multi_to_main_token = multi_to_main_token
                    
    def to_json(self):
        info_json = {
            "initial": self.init_tokens,
            "tagged_lemma": self.tagged_lemma,
            "multi": [self.multitokens, self.multi_to_main_token]
        }
        
        if self.deeppavlov_lemma is not None:
            info_json["deeppavlov"] = self.deeppavlov_lemma
            
        if self.deeppavlov_pos is not None:
            info_json["pos"] = self.deeppavlov_pos
            
        if self.syntax_tree:
            info_json["syntax"] = self.syntax_tree.to_json()
            
        return info_json


class SentenceReader:
    def __init__(self, thesaurus, need_deeppavlov=True, deeppavlov_model=None,
                 need_syntax=True, syntax_model=None):
        self.need_deeppavlov = need_deeppavlov
        
        if need_deeppavlov:
            self.deeppavlov_lemma = deeppavlov_model if deeppavlov_model else build_model(
                configs.morpho_tagger.BERT.morpho_ru_syntagrus_bert,
                download=False
            )
            
        if need_syntax:
            self.syntax_model = syntax_model if syntax_model else build_model(
                configs.syntax.syntax_ru_syntagrus_bert,
                download=False
            )
        else:
            self.syntax_model = None
        
        self.tokenizer = WordPunctTokenizer()
        self.thesaurus = thesaurus
                
    def process_file(self, filename, verbose=False):
        tagged_lemmas = []
        initial_sentences = []
        
        # Stats for output
        broken_sentences = 0
        failed_lemmatize = 0
        
        with open(filename) as tagged_file:
            current_sentence_tokens = []
            current_sentence_lemmas = []
            need_append = False

            for line in tagged_file.readlines():
                if line.startswith("# sent_id"):
                    need_append = True
                elif line.startswith("# text"):
                    continue
                elif len(line) < 2:
                    sentences_lemma_divided = self.divide_tagged(current_sentence_lemmas)
                    sentence_initial_divided = self.divide_tagged(current_sentence_tokens)
                    
                    tagged_lemmas += sentences_lemma_divided
                    initial_sentences += sentence_initial_divided
                    broken_sentences += (len(sentences_lemma_divided) - 1)
                    
                    need_append = False
                    current_sentence_tokens = []
                    current_sentence_lemmas = []
                else:
                    if need_append:
                        line_splitted = line.split('\t')
                        current_sentence_tokens.append(line_splitted[1].lower())
                        current_sentence_lemmas.append(line_splitted[2].lower())
                        
        parsed_sentences = []
        
        for init_tokens, lemma_tokens in zip(initial_sentences, tagged_lemmas):
            deeppavlov_lemma = None
            deeppavlov_pos = None
            
            if self.need_deeppavlov:
                try:
                    deeppavlov_lemma, deeppavlov_pos = self.get_deeppavlov_info(init_tokens)
                except:
                    failed_lemmatize += 1
                    deeppavlov_lemma = None
                    deeppavlov_pos = None
            
            parsed_sentences.append(ParsedSentence(
                init_tokens,
                lemma_tokens,
                self.thesaurus,
                deeppavlov_lemma,
                deeppavlov_pos,
                self.syntax_model
            ))
        
        if verbose:
            print("Processed {}. Recovered {} sentences, lost {} too long".format(
                filename,
                broken_sentences,
                failed_lemmatize
            ))
        
        return parsed_sentences
    
    def process_directory(self, dir_path, verbose=False):
        text_names = listdir(dir_path)
        all_sentences = []
        
        for filename in text_names:
            full_path = join(dir_path, filename)
            parsed_sentences = self.process_file(full_path)
            all_sentences += parsed_sentences
            
        return all_sentences
       
    def divide_tagged(self, tagged_sentence):
        single_sentence = " ".join(tagged_sentence)
        sentence_parts = single_sentence.split(".")
        return [self.tokenizer.tokenize(part) + ["."] for part in sentence_parts if len(part) > 0]
        
    def get_deeppavlov_info(self, tagged_sentence):
        sentences = [tagged_sentence]
        morpho_tokens = self.deeppavlov_lemma(sentences)[0].split('\n')
        splitted_info = [x.split('\t') for x in morpho_tokens]
        lemmatized_tokens = [splitted[2] for splitted in splitted_info if len(splitted) == 10]
        pos = [splitted[3] for splitted in splitted_info if len(splitted) == 10]
        return lemmatized_tokens, pos
    
    
def make_jsons_from_directory(reader, dir_from, dir_to, suffix="_processed", encoding="utf8"):
    text_names = listdir(dir_from)
    
    for filename in text_names:
        path_from = join(dir_from, filename)
        path_to = join(dir_to, filename + suffix + ".json")
        
        if os.path.isfile(path_to):
            continue
            
        parsed_sentences = reader.process_file(path_from)
        parsed_sentences = [sent.to_json() for sent in parsed_sentences]
        with open(path_to, "w", encoding=encoding) as out_file:
            json.dump(parsed_sentences, out_file)