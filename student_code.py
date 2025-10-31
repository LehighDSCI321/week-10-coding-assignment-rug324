"""
Implements SortableDigraph → TraversableDigraph → DAG
with DFS, BFS, topological sorting, and cycle–safe DAG edge insertion.
"""

from collections import deque


class SortableDigraph:
    """Base directed graph class supporting sorting and weighted edges."""

    def __init__(self):
        self.nodes = {}   # node -> value
        self.edges = {}   # node -> {neighbor: weight}

    def add_node(self, name, value=None):
        """Add a node with an optional value."""
        self.nodes[name] = value
        if name not in self.edges:
            self.edges[name] = {}

    def add_edge(self, u, v, edge_weight=None):
        """Add an edge u → v with optional weight."""
        self.add_node(u)
        self.add_node(v)
        self.edges[u][v] = edge_weight

    def get_nodes(self):
        """Return all nodes in sorted order."""
        return sorted(self.nodes.keys())

    def get_node_value(self, name):
        """Return the stored value for a node."""
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' does not exist.")
        return self.nodes[name]

    def get_edge_weight(self, u, v):
        """Return stored weight for edge u → v."""
        try:
            return self.edges[u][v]
        except KeyError as exc:
            raise KeyError(f"Edge '{u} → {v}' does not exist.") from exc

    def successors(self, name):
        """Return sorted list of direct successors."""
        return sorted(self.edges.get(name, {}).keys())

    def predecessors(self, name):
        """Return sorted list of direct predecessors."""
        preds = [src for src, nbrs in self.edges.items() if name in nbrs]
        return sorted(preds)

    def top_sort(self):
        """Perform topological sort; raise ValueError if a cycle exists."""
        indegree = {n: 0 for n in self.nodes}
        for nbrs in self.edges.values():
            for child in nbrs:
                indegree[child] += 1

        queue = deque(sorted([n for n, d in indegree.items() if d == 0]))
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in sorted(self.edges.get(node, {}).keys()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Graph has a cycle; cannot topologically sort.")
        return order


class TraversableDigraph(SortableDigraph):
    """Adds DFS and BFS traversal."""

    def dfs(self, start):
        """Depth-first search traversal excluding start node."""
        visited = set()
        result = []

        def _dfs(node):
            for nxt in sorted(self.edges.get(node, {}).keys()):
                if nxt not in visited:
                    visited.add(nxt)
                    result.append(nxt)
                    _dfs(nxt)

        _dfs(start)
        return result

    def bfs(self, start):
        """Breadth-first search traversal excluding start node."""
        seen = {start}
        queue = deque()

        for nxt in sorted(self.edges.get(start, {}).keys()):
            seen.add(nxt)
            queue.append(nxt)

        while queue:
            node = queue.popleft()
            yield node
            for nxt in sorted(self.edges.get(node, {}).keys()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)


class DAG(TraversableDigraph):
    """Directed Acyclic Graph; disallows edges that create cycles."""

    def add_edge(self, u, v, edge_weight=None):
        """Add edge only if it does not introduce a cycle."""
        self.add_node(u)
        self.add_node(v)

        # Check for potential cycle (path from v to u already exists)
        if self._path_exists(v, u):
            raise ValueError(f"Adding edge {u} → {v} creates a cycle.")

        self.edges[u][v] = edge_weight

    def _path_exists(self, start, goal):
        """Return True if there exists a path start → goal."""
        stack = [start]
        visited = set()
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            if node not in visited:
                visited.add(node)
                for nxt in self.edges.get(node, {}).keys():
                    if nxt not in visited:
                        stack.append(nxt)
        return False
