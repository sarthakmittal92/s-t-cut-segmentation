from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network

class BK:
    
    def __init__(self, G, s, t, capacity='capacity', residual=None, value_only=False, cutoff=None):
        self.G = G
        self.s = s
        self.t = t
        self.capacity = capacity
        self.residual = residual
        self.value_only = value_only
        self.cutoff = cutoff
        self.BKImplementation()
        self.R.graph['algorithm'] = 'boykov_kolmogorov'
    
    def getResidual (self):
        return self.R
    
    def BKImplementation (self):
        if self.s not in self.G:
            raise nx.NetworkXError(f'Source node {s} not in graph')
        elif self.t not in self.G:
            raise nx.NetworkXError(f'Sink node {t} not in graph')
        elif self.s == self.t:
            raise nx.NetworkXError(f'Source and sink node are the same')
        if self.residual is None:
            self.R = build_residual_network(self.G, self.capacity)
        else:
            self.R = self.residual
        for u in self.R:
            for e in self.R[u].values():
                e['flow'] = 0
        self.INFTY = self.R.graph['inf']
        if self.cutoff is None:
            self.cutoff = self.INFTY
        self.successor = self.R.succ
        self.predecessor = self.R.pred
        self.source_tree = {self.s: None}
        self.target_tree = {self.t: None}
        self.active = deque([self.s, self.t])
        self.orphans = deque()
        flow_value = 0
        self.time = 1
        self.timestamp = {self.s: self.time, self.t: self.time}
        self.dist = {self.s: 0, self.t: 0}
        while flow_value < self.cutoff:
            u, v = self.grow()
            if u is None:
                break
            self.time += 1
            flow_value += self.augment(u, v)
            self.adopt()
        if flow_value * 2 > self.INFTY:
            raise nx.NetworkXUnbounded('INFTY capacity path, flow unbounded above.')

        self.R.graph['trees'] = (self.source_tree, self.target_tree)
        self.R.graph['flow_value'] = flow_value
        
    def grow (self):
        while self.active:
            u = self.active[0]
            if u in self.source_tree:
                this_tree = self.source_tree
                other_tree = self.target_tree
                neighbors = self.successor
            else:
                this_tree = self.target_tree
                other_tree = self.source_tree
                neighbors = self.predecessor
            for v, attr in neighbors[u].items():
                if attr['capacity'] - attr['flow'] > 0:
                    if v not in this_tree:
                        if v in other_tree:
                            return (u, v) if this_tree is self.source_tree else (v, u)
                        this_tree[v] = u
                        self.dist[v] = self.dist[u] + 1
                        self.timestamp[v] = self.timestamp[u]
                        self.active.append(v)
                    elif v in this_tree and self.isCloser(u, v):
                        this_tree[v] = u
                        self.dist[v] = self.dist[u] + 1
                        self.timestamp[v] = self.timestamp[u]
            _ = self.active.popleft()
        return None, None
    
    def augment (self, u,v):
        attr = self.successor[u][v]
        flow = min(self.INFTY, attr['capacity'] - attr['flow'])
        path = [u]
        w = u
        while w != self.s:
            n = w
            w = self.source_tree[n]
            attr = self.predecessor[n][w]
            flow = min(flow, attr['capacity'] - attr['flow'])
            path.append(w)
        path.reverse()
        path.append(v)
        w = v
        while w != self.t:
            n = w
            w = self.target_tree[n]
            attr = self.successor[n][w]
            flow = min(flow, attr['capacity'] - attr['flow'])
            path.append(w)
        it = iter(path)
        u = next(it)
        these_orphans = []
        for v in it:
            self.successor[u][v]['flow'] += flow
            self.successor[v][u]['flow'] -= flow
            if self.successor[u][v]['flow'] == self.successor[u][v]['capacity']:
                if v in self.source_tree:
                    self.source_tree[v] = None
                    these_orphans.append(v)
                if u in self.target_tree:
                    self.target_tree[u] = None
                    these_orphans.append(u)
            u = v
        self.orphans.extend(sorted(these_orphans, key=self.dist.get))
        return flow
    
    def adopt (self):
        while self.orphans:
            u = self.orphans.popleft()
            if u in self.source_tree:
                tree = self.source_tree
                neighbors = self.predecessor
            else:
                tree = self.target_tree
                neighbors = self.successor
            nbrs = ((n, attr, self.dist[n]) for n, attr in neighbors[u].items()
                    if n in tree)
            for v, attr, d in sorted(nbrs, key=itemgetter(2)):
                if attr['capacity'] - attr['flow'] > 0:
                    if self.hasValidRoot(v, tree):
                        tree[u] = v
                        self.dist[u] = self.dist[v] + 1
                        self.timestamp[u] = self.time
                        break
            else:
                nbrs = ((n, attr, self.dist[n]) for n, attr in neighbors[u].items()
                        if n in tree)
                for v, attr, d in sorted(nbrs, key=itemgetter(2)):
                    if attr['capacity'] - attr['flow'] > 0:
                        if v not in self.active:
                            self.active.append(v)
                    if tree[v] == u:
                        tree[v] = None
                        self.orphans.appendleft(v)
                if u in self.active:
                    self.active.remove(u)
                del tree[u]
    
    def hasValidRoot (self, n, tree):
        path = []
        v = n
        while v is not None:
            path.append(v)
            if v == self.s or v == self.t:
                base_dist = 0
                break
            elif self.timestamp[v] == self.time:
                base_dist = self.dist[v]
                break
            v = tree[v]
        else:
            return False
        length = len(path)
        for i, u in enumerate(path, 1):
            self.dist[u] = base_dist + length - i
            self.timestamp[u] = self.time
        return True
    
    def isCloser (self, u,v):
        return self.timestamp[v] <= self.timestamp[u] and self.dist[v] > self.dist[u] + 1