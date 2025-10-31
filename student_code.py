"""
Alternative implementation of SortableDigraph → TraversableDigraph → DAG
with identical functionality but different structure and variable naming.
"""

from collections import deque

class SortableDigraph:
    """Directed graph with sortable nodes and weighted edges."""

    def __init__(self):
        self.nodes = {}   # node -> value
        self.edges = {}   # node -> {neighbor: weight}

    def add_node(self, name, value=None):
        """Add node with optional stored value."""
        if name not in self.nodes:
            self.nodes[name] = value
        if name not in self.edges:
            self.edges[name] = {}

    def add_edge(self, start, end, weight=None):
        """Add edge start → end with optional weight."""
        self.add_node(start)
        self.add_node(end)
        self.edges[start][end] = weight

    def get_nodes(self):
        """Return all nodes in sorted order."""
        return sorted(self.nodes.keys())

    def get_node_value(self, name):
        """Return stored value for node."""
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' does not exist.")
        return self.nodes[name]

    def get_edge_weight(self, start, end):
        """Return weight for edge start → end."""
        try:
            return self.edges[start][end]
        except KeyError:
            raise KeyError(f"Edge '{start} → {end}' does not exist.")

    def successors(self, name):
        """Return sorted list of direct successors."""
        return sorted(self.edges.get(name, {}).keys())

    def predecessors(self, name):
        """Return sorted list of direct predecessors."""
        preds = [u for u, nbrs in self.edges.items() if name in nbrs]
        return sorted(preds)

    def top_sort(self):
        """Perform topological sort. Raise ValueError if cycle exists."""
        indeg = {n: 0 for n in self.nodes}
        for u, nbrs in self.edges.items():
            for v in nbrs:
                indeg[v] += 1

        queue = deque(sorted([n for n, d in indeg.items() if d == 0]))
        order = []

        while queue:
            n = queue.popleft()
            order.append(n)
            for child in sorted(self.edges.get(n, {})):
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Graph has a cycle; cannot topologically sort.")
        return order

class TraversableDigraph(SortableDigraph):
    """Extends SortableDigraph with DFS and BFS traversal."""

    def dfs(self, start):
        """Depth-first traversal excluding the start node itself."""
        visited, order = set(), []

        def explore(node):
            for nxt in sorted(self.edges.get(node, {})):
                if nxt not in visited:
                    visited.add(nxt)
                    order.append(nxt)
                    explore(nxt)

        explore(start)
        return order

    def bfs(self, start):
        """Breadth-first traversal excluding the start node."""
        seen = {start}
        q = deque()

        for nxt in sorted(self.edges.get(start, {})):
            seen.add(nxt)
            q.append(nxt)

        while q:
            node = q.popleft()
            yield node
            for nxt in sorted(self.edges.get(node, {})):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)

class DAG(TraversableDigraph):
    """Directed Acyclic Graph ensuring no cycle creation."""

    def add_edge(self, u, v, weight=None):
        """Add edge only if it does not create a cycle."""
        self.add_node(u)
        self.add_node(v)

        if self._path_exists(v, u):
            raise ValueError(f"Adding edge {u} → {v} creates a cycle.")

        self.edges[u][v] = weight

    def _path_exists(self, start, goal):
        """Check if a path exists from start → goal."""
        stack, visited = [start], set()
