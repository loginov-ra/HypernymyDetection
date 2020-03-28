import json
from deeppavlov import build_model, configs


PARENT_POS = 6
ROLE_POS = 7
TOKEN_POS = 1


class SyntaxTree:
    class Node:
        def __init__(self, idx, parent_idx, role, token):
            self.idx = idx
            self.parent_idx = parent_idx
            self.role = role
            self.token = token
            
            self.token_inconsistency = 0
            
        def __str__(self):
            return u"{};{};{};{}".format(
                self.idx,
                self.parent_idx,
                self.role,
                self.token
            )
    
    def __init__(self, model=None, download_model=False):
        if model is None:
            self.model = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=download_model)
        else:
            self.model = model
        
    def load(self, tokens, multitokens=None):
        self.tokens = tokens
        self.multitokens = multitokens if multitokens else tokens
    
    def __str__(self):
        return json.dumps(self.to_json(), ensure_ascii=False)
            
    def build(self):
        parsed = self.model([self.tokens])[0].split('\n')[:-1]
        self.nodes = []
        
        for i, token_info in enumerate(parsed):
            token_info = token_info.split('\t')
            node = SyntaxTree.Node(
                idx=i,
                parent_idx=int(token_info[PARENT_POS])-1,
                role=token_info[ROLE_POS],
                token=self.tokens[i]
            )
            self.nodes.append(node)
            if self.tokens[i] != token_info[TOKEN_POS]:
                self.token_inconsistency += 1
            
        self.size = len(self.nodes)
        self.children = [[] for _ in range(self.size)]
        
        for i, node in enumerate(self.nodes):
            if node.role == "root":
                self.root = node.idx
            self.children[node.parent_idx].append(node.idx)
            
    def to_json(self):
        return [str(node) for node in self.nodes]