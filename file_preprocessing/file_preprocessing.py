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


class ParsedSentence:
    def __init__(self, tagged_lemma, mwe_tokenizer, deeppavlov_lemma=None):
        self.tagged_lemma = tagged_lemma
        self.deeppavlov_lemma = deeppavlov_lemma
        self.mwe_tokenizer = mwe_tokenizer
    
    
    def make_multitokens(self):
        tokens = self.tagged_lemma if self.deeppavlov_lemma is None else self.deeppavlov_lemma
        return self.mwe_tokenizer.tokenize(tokens)
    
    
    def to_json(self):
        info_json = {
            "initial": self.tagged_lemma,
            "multi": self.make_multitokens()
        }
        
        if self.deeppavlov_lemma is not None:
            info_json["deeppavlov"] = self.deeppavlov_lemma
            
        return info_json


class SentenceReader:
    def __init__(self, thesaurus, need_deeppavlov=True):
        self.need_deeppavlov = need_deeppavlov
        
        if need_deeppavlov:
            self.deeppavlov_lemma = build_model(
                configs.morpho_tagger.BERT.morpho_ru_syntagrus_bert,
                download=False
            )
        
        self.tokenizer = WordPunctTokenizer()
        self.mwe_tokenizer = thesaurus.form_mwe_tokenizer()
        
        
    def process_file(self, filename, verbose=False):
        tagged_lemmas = []
        
        # Stats for output
        broken_sentences = 0
        failed_lemmatize = 0
        
        with open(filename) as tagged_file:
            current_sentence_tokens = []
            need_append = False

            for line in tagged_file.readlines():
                if line.startswith("# sent_id"):
                    need_append = True
                elif line.startswith("# text"):
                    continue
                elif len(line) < 2:
                    sentences_divided = self.divide_tagged(current_sentence_tokens)
                    tagged_lemmas += sentences_divided
                    broken_sentences += (len(sentences_divided) - 1)
                    
                    need_append = False
                    current_sentence_tokens = []
                else:
                    if need_append:
                        current_sentence_tokens.append(line.split('\t')[2])
                        
        parsed_sentences = []
        
        for tokens in tagged_lemmas:
            deeppavlov_lemma = None
            
            if self.need_deeppavlov:
                try:
                    deeppavlov_lemma = self.get_deeppavlov_lemma(tokens)
                except:
                    failed_lemmatize += 1
                    deeppavlov_lemma = None
            
            parsed_sentences.append(ParsedSentence(
                tokens,
                self.mwe_tokenizer,
                deeppavlov_lemma
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
        if tagged_sentence[-1] == ".":
            tagged_sentence = tagged_sentence[:-1]
            
        if "." not in tagged_sentence:
            return [tagged_sentence + ["."]]
        
        full_text = " ".join(tagged_sentence)
        sentence_parts = full_text.split(".")
        
        return [self.tokenizer.tokenize(part) + ["."] for part in sentence_parts]
    
    
    def get_deeppavlov_lemma(self, tagged_sentence):
        sent = " ".join(tagged_sentence)
        sentences = [sent]
        morpho_tokens = self.deeppavlov_lemma(sentences)[0].split('\n')
        lemmatized_tokens = [x.split('\t')[2] for x in morpho_tokens if len(x.split('\t')) == 10]
        return lemmatized_tokens
    
    
def make_jsons_from_directory(reader, dir_from, dir_to, suffix="_processed", encoding="utf8"):
    text_names = listdir(dir_from)
    
    for filename in text_names:
        path_from = join(dir_from, filename)
        path_to = join(dir_to, filename + suffix + ".json")
        parsed_sentences = reader.process_file(path_from)
        parsed_sentences = [sent.to_json() for sent in parsed_sentences]
        with open(path_to, "w", encoding=encoding) as out_file:
            json.dump(parsed_sentences, out_file)