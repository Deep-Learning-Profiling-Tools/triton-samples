"""Test pickability of Triton compilation errors."""

import triton
import pickle


class Node:
    """Represents a node with line and column information."""

    def __init__(self, lineno: int, col_offset: int):
        self.lineno = lineno
        self.col_offset = col_offset


# Test serialization
src = ""
node = Node(0, 0)
error = triton.CompilationError(src, node)

pickled = pickle.dumps(error)
origin = pickle.loads(pickled)
print(origin)
