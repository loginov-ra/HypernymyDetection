import json
from deeppavlov import build_model, configs
from queue import Queue

PARENT_POS = 6
ROLE_POS = 7
TOKEN_POS = 1

ADDITIONAL_WORDS = [
    'который',
    'должный',
    'также',
    'как',
    'тот',
    'весь',
    'другой',
    'такой',
    'несколько',
    'и',
    'или',
    'никакой',
    'же',
    'ряд',
    'так',
    'а',
]

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
        self.children_roles = [dict() for _ in range(self.size)]
        self.children_types = [dict() for _ in range(self.size)]
        
        for i, node in enumerate(self.nodes):
            if node.role == "root" or node.parent_idx == -1:
                self.root = node.idx
                continue
                
            self.children[node.parent_idx].append(node.idx)
            self.children_roles[node.parent_idx][node.idx] = node.role
            self.children_types[node.parent_idx][node.idx] = 'usual'
            
            self.children[node.idx].append(node.parent_idx)
            self.children_roles[node.idx][node.parent_idx] = 'parent'
            self.children_types[node.idx][node.parent_idx] = 'parent'
            
    
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
    
    def _compress_conj_edges_dfs(self, idx):
        self._used[idx] = True
        
        for c_idx in self.children[idx]:
            if not self._used[c_idx]:
                self._compress_conj_edges_dfs(c_idx)
                
        for c_idx in self.children[idx]:
            role = self.children_roles[idx][c_idx]
            if role == 'parent':
                continue
            for new_child in self.children[c_idx]:
                new_role = self.children_roles[c_idx][new_child]
                if new_role == 'conj':
                    self.children[idx].append(new_child)
                    self.children[new_child].append(idx)
                    self.children_types[idx][new_child] = 'conj'
                    self.children_roles[idx][new_child] = role
                    self.children_types[new_child][idx] = 'parent'
                    self.children_roles[new_child][idx] = 'parent'
                        
    def compress_conj_edges(self):
        self._used = [False] * len(self.nodes)
        self._compress_conj_edges_dfs(self.root)
        del self._used
    
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
            
    def get_syntax_pattern(self, v_from, v_to, pos, lemmas, additional_words=ADDITIONAL_WORDS):
        path = self.get_shortest_path(v_from, v_to)
        edge_descriptions = []
        
        if path is None:
            return None
        
        for from_child in self.children[v_from]:
            if lemmas[from_child] in additional_words and from_child not in path:
                path = [from_child] + path
                break
                
        for to_child in self.children[v_to]:
            if lemmas[to_child] in additional_words and to_child not in path:
                path = path + [to_child]
                break
        
        for i, edge_start in enumerate(path[:-1]):
            edge_finish = path[i + 1]
            if self.children_roles[edge_start][edge_finish] == 'parent':
                role = self.children_roles[edge_finish][edge_start]
            elif self.children_roles[edge_finish][edge_start] == 'parent':
                role = self.children_roles[edge_start][edge_finish]
            else:
                raise(ValueError("Role was not found for tree: {}".format(self.__str__())))
            
            edge_description = "{start_word}:{start_pos}:{role}:{finish_pos}:{finish_word}".format(
                start_word=lemmas[edge_start] if edge_start not in [v_from, v_to] else "{}",
                start_pos=pos[edge_start],
                role=role,
                finish_pos=pos[edge_finish],
                finish_word=lemmas[edge_finish] if edge_finish not in [v_from, v_to] else "{}"
            )
            edge_descriptions.append(edge_description)
            
        return edge_descriptions
        
    def to_json(self):
        return [str(node) for node in self.nodes]