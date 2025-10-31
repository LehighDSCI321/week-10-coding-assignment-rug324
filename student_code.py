"""
student_code.py

Implements SortableDigraph → TraversableDigraph → DAG
with DFS, BFS, topological sorting, and cycle–safe DAG edge insertion.
"""

from collections import deque

class SortableDigraph:
    """
    Base digraph class storing nodes with values and weighted edges.
    """

    def __init__(self):
        self.nodes = {}                 # {node: value}
        self.edges = {}                 # {node: {child: weight}}

    def add_node(self, name, value=None):
        """
        Add a node with optional value.
        """
        self.nodes[name] = value
        if name not in self.edges:
            self.edges[name] = {}

    def add_edge(self, u, v, edge_weight=None):
        """
        Add edge u → v with optional weight.
        """
        self.add_node(u)
        self.add_node(v)
        self.edges[u][v] = edge_weight

    def get_nodes(self):
        """
        Return sorted list of node names.
        """
        return sorted(self.nodes.keys())

    def get_node_value(self, name):
        """
        Return stored value for node.
        """
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' does not exist.")
        return self.nodes[name]

    def get_edge_weight(self, u, v):
        """
        Return stored weight for edge u → v.
        """
        if u not in self.edges or v not in self.edges[u]:
            raise KeyError(f"Edge '{u} → {v}' does not exist.")
        return self.edges[u][v]

    def successors(self, name):
        """
        Return sorted list of direct successors of name.
        """
        if name not in self.edges:
            return []
        return sorted(self.edges[name].keys())

    def predecessors(self, name):
        """
        Return sorted direct predecessors of name.
        """
        result = []
        for p, children in self.edges.items():
            if name in children:
                result.append(p)
        return sorted(result)

    def top_sort(self):
        """
        Topological sort.
        """
        indegree = {n: 0 for n in self.nodes}

        # compute indegree
        for children in self.edges.values():
            for child in children:
                indegree[child] += 1

        queue = deque(sorted([n for n, d in indegree.items() if d == 0]))
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for child in sorted(self.edges.get(node, {}).keys()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(result) != len(self.nodes):
            raise ValueError("Graph has a cycle; cannot topologically sort.")

        return result


class TraversableDigraph(SortableDigraph):
    """
    Adds DFS / BFS traversal.
    """

    def dfs(self, start):
        """
        DFS excluding start.
        """
        visited = set()
        order = []

        def _visit(node):
            for nbr in sorted(self.edges.get(node, {}).keys()):
                if nbr not in visited:
                    visited.add(nbr)
                    order.append(nbr)
                    _visit(nbr)

        _visit(start)
        return order

    def bfs(self, start):
        """
        BFS excluding start.
        """
        visited = {start}
        queue = deque()

        for nbr in sorted(self.edges.get(start, {}).keys()):
            visited.add(nbr)
            queue.append(nbr)

        while queue:
            node = queue.popleft()
            yield node
            for nbr in sorted(self.edges.get(node, {}).keys()):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)


class DAG(TraversableDigraph):
    """
    DAG that prevents cycle creation.
    """

    def add_edge(self, u, v, edge_weight=None):
        """
        Add edge if no cycle; else raise ValueError.
        """
        self.add_node(u)
        self.add_node(v)

        if self._has_path(v, u):
            raise ValueError(f"Adding edge {u} → {v} creates a cycle.")

        self.edges[u][v] = edge_weight

    def _has_path(self, start, goal):
        """
        True if path start → ... → goal exists.
        """
        stack = [start]
        visited = set()

        while stack:
            node = stack.pop()
            if node == goal:
                return True
            if node not in visited:
                visited.add(node)
                for nbr in self.edges.get(node, {}).keys():
                    stack.append(nbr)

        return False
