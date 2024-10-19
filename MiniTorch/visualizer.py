import graphviz as gv
from collections import deque

def trace(root):
    queue = deque([root])
    visited = set()

    nodes = set()
    edges = set()

    while queue:
        node = queue.pop()
        nodes.add(node)

        for child in node._prev:
            edges.add((child, node))

            if child not in visited:
                queue.append(child)

    return nodes, edges

def build_graph(nodes, edges, format = "svg", rankdir = "LR"):
    dot = gv.Digraph(format = format, graph_attr = {"rankdir": rankdir})

    for node in nodes:
        name = str(id(node))
        label = f"{node.data} | {node.grad}"
        dot.node(name = name, label = label, shape = "record")

        if node._op:
            dot.node(name = name + node._op, label = f"{node._op}")
            dot.edge(name + node._op, name)

    for edge in edges:
        a, b = edge
        dot.edge(str(id(a)), str(id(b)) + b._op)

    return dot
