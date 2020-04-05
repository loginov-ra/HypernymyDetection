import json
from deeppavlov import build_model, configs
from queue import Queue

PARENT_POS = 6
ROLE_POS = 7
TOKEN_POS = 1


class SyntaxTree:
    class Node:
        def __init__(self, idx=-1, parent_idx=-1, role="empty", token="empty"):
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
        
        def load_from_string(self, node_str):
            fields = node_str.split(";")
            self.idx = int(fields[0])
            self.parent_idx = int(fields[1])
            self.role = fields[2]
            self.token = fields[3]
    
    def __init__(self, model=None, download_model=False, empty=False):
        if empty:
            return
        
        if model is None:
            self.model = build_model(configs.syntax.syntax_ru_syntagrus_bert, download=download_model)
        else:
            self.model = model
        
    def load(self, tokens, multitokens=None):
        self.tokens = tokens
        self.multitokens = multitokens if multitokens else tokens
    
    def __str__(self):
        return json.dumps(self.to_json(), ensure_ascii=False)
    
    def build_additional_info_from_nodes(self):
        self.size = len(self.nodes)
        self.children = [[] for _ in range(self.size)]
        
        for i, node in enumerate(self.nodes):
            if node.role == "root":
                self.root = node.idx
            self.children[node.parent_idx].append(node.idx)
            self.children[node.idx].append(node.parent_idx)
    
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
            
        self.build_additional_info_from_nodes()
            
    def load_from_json(self, nodes_json):
        self.nodes = []
        for node_str in nodes_json:
            new_node = SyntaxTree.Node()
            new_node.load_from_string(node_str)
            self.nodes.append(new_node)
        self.build_additional_info_from_nodes()
            
    def get_shortest_path(self, v_from, v_to):
        q = Queue()
        q.put(v_from)
        prev_vertex = [-1] * self.size
        used = [False] * self.size
        used[v_from] = True
        
        while not q.empty():
            new_vertex = q.get()
            for neighbour in self.children[new_vertex]:
                if not used[neighbour]:
                    prev_vertex[neighbour] = new_vertex
                    q.put(neighbour)
                    used[neighbour] = True
                    
        if prev_vertex[v_to] == -1:
            return None
        
        curr_vertex = v_to
        path = [v_to]
        
        while curr_vertex != v_from:
            curr_vertex = prev_vertex[curr_vertex]
            path.append(curr_vertex)
            
        return path[-1::-1]
            
    def to_json(self):
        return [str(node) for node in self.nodes]