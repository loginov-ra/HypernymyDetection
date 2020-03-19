import json
import sys
import os
from os import listdir
from os.path import join
from lxml import etree
from collections import defaultdict


class ThesaurusParser:
    def read_entities_from_xml(self):
        entities_path = join(self.thesaurus_dir, "concepts.xml")
        
        with open(entities_path) as xml_file:
            xml = xml_file.read()
        
        id_to_entity = dict()
        entity_to_id = dict()
        
        root = etree.fromstring(xml)
        
        for entity in root.getchildren():
            name = entity.find("name").text
            idx = entity.get("id")
            id_to_entity[idx] = name
            entity_to_id[name] = idx
            
        self.id_to_entity = id_to_entity
        self.entity_to_id = entity_to_id
    
    
    def read_entities_hypernymy(self):
        relations_path = join(self.thesaurus_dir, "relations.xml")
    
        with open(relations_path) as xml_file:
            xml = xml_file.read()
            
        entity_hypernyms = defaultdict(list)
        
        root = etree.fromstring(xml)
        
        for relation in root.getchildren():
            rel_type = relation.get("name")
            
            if rel_type == "ВЫШЕ":
                lhs = relation.get("from")
                rhs = relation.get("to")
                entity_hypernyms[lhs].append(rhs) 
                
        self.entity_hypernyms = dict(entity_hypernyms)
    
    
    def read_text_entries(self):
        text_entries_path = join(self.thesaurus_dir, "text_entry.xml")
        
        with open(text_entries_path) as xml_file:
            xml = xml_file.read()
        
        text_entries = dict()
        lemma_to_entry = dict()
        
        root = etree.fromstring(xml)
        
        for entry in root.getchildren():
            idx = entry.get("id")
            name = entry.find("name").text.lower()
            lemma = entry.find("lemma").text.lower()
            main_word = entry.find("main_word").text
            
            if main_word is None:
                main_word = lemma
            else:
                main_word = main_word.lower()
            
            entry_dict = {"name": name, "lemma": lemma, "main_word": main_word}
            
            text_entries[idx] = entry_dict
            lemma_to_entry[lemma] = idx
    
        self.text_entries = text_entries
        self.lemma_to_entry = lemma_to_entry
    
    def build_synsets(self):
        synsets_path = join(self.thesaurus_dir, "synonyms.xml")
        
        with open(synsets_path) as xml_file:
            xml = xml_file.read()
            
        synsets_dict = defaultdict(list)
        
        root = etree.fromstring(xml)
        
        for entry in root.getchildren():
            entity_id = entry.get("concept_id")
            entry_id = entry.get("entry_id")
            synsets_dict[entity_id].append(entry_id)
        
        self.synsets_dict = dict(synsets_dict)
        
        
    def make_first_level_hypernymy(self):
        hypernyms_dict = defaultdict(set)
        
        for hyponym_entity, hypernym_entities in self.entity_hypernyms.items():
            for hypernym_entity in hypernym_entities:
                for hyponym_entry_id in self.synsets_dict[hyponym_entity]:
                    for hypernym_entry_id in self.synsets_dict[hypernym_entity]:
                        hyponym_lemma = self.text_entries[hyponym_entry_id]["lemma"]
                        hypernym_lemma = self.text_entries[hypernym_entry_id]["lemma"]
                        hypernyms_dict[hyponym_lemma].add(hypernym_lemma)
                    
        self.hypernyms_dict = dict([(hyponym, list(hypernyms)) 
                                    for hyponym, hypernyms in hypernyms_dict.items()])

    
    def closure_dfs(self, lemma):
        self.closure_built[lemma] = True
        self.closed_hypernymy[lemma].update(self.hypernyms_dict[lemma])
        self.visited += 1
        
        if self.verbose:
            print("Visited:", self.visited)
        
        for hypernym_lemma in self.hypernyms_dict[lemma]:
            if hypernym_lemma not in self.hypernyms_dict:
                continue
            
            if not self.closure_built[hypernym_lemma]:
                self.closure_dfs(hypernym_lemma)
            
            self.closed_hypernymy[lemma].update(self.closed_hypernymy[hypernym_lemma])
                
        self.closed_hypernymy[lemma] = list(self.closed_hypernymy[lemma])
        
    
    def __init__(self, thesaurus_dir=None, need_closure=True, verbose=False):
        if thesaurus_dir is None:
            return
        
        self.verbose = verbose
        self.thesaurus_dir = thesaurus_dir
        
        # Parsing thesaurus files
        self.read_entities_from_xml()
        self.read_entities_hypernymy()
        self.read_text_entries()
        self.build_synsets()
        
        # Building hypernymy dictionary
        self.make_first_level_hypernymy()
        
        if need_closure:
            self.closed_hypernymy = defaultdict(set)
            self.closure_built = dict([(lemma, False) for lemma in self.hypernyms_dict])
            self.visited = 0
            for lemma in self.hypernyms_dict:
                if not self.closure_built[lemma]:
                    self.closure_dfs(lemma)
            self.closed_hypernymy = dict(self.closed_hypernymy)
            
          
    def save_thesaurus(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            raise ValueError("Path {} already exists!".format(path))

        path_one_level = join(path, "one_level_hypernyms.json")
        path_closed = join(path, "closed_hypernyms.json")
        path_text_entries = join(path, "text_entries.json")
        path_lemma_to_entry = join(path, "lemma_to_entry.json")

        paths = [path_one_level, path_closed,
                 path_text_entries, path_lemma_to_entry]

        dicts = [parser.hypernyms_dict, parser.closed_hypernymy,
                 parser.text_entries, parser.lemma_to_entry]

        for dict_path, dictionary in zip(paths, dicts):
            json.dump(dictionary, open(dict_path, "w"))
            
            
    def load_thesaurus(self, path):
        path_one_level = join(path, "one_level_hypernyms.json")
        path_closed = join(path, "closed_hypernyms.json")
        path_text_entries = join(path, "text_entries.json")
        path_lemma_to_entry = join(path, "lemma_to_entry.json")
        
        self.hypernyms_dict = json.load(open(path_one_level))
        self.closed_hypernymy = json.load(open(path_closed))
        self.text_entries = json.load(open(path_text_entries))
        self.lemma_to_entry = json.load(open(path_lemma_to_entry))