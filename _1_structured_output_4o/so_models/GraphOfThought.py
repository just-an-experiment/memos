# A representation of a thought process or idea flow through interconnected nodes and edges

from pydantic import BaseModel

class Node(BaseModel):
    id: int
    description: str
    tag: str
    
    class Config:
        title = "Node"
        description = "A single node in the graph of thought"
    id: int
    description: str
    tag: str

class Edge(BaseModel):
    source_node_id: int
    target_node_id: int
    relationship_description: str
    
    class Config:
        title = "Edge"
        description = "A connection or relationship between two nodes in the graph of thought"

class GraphOfThought(BaseModel):
    nodes: list[Node]
    edges: list[Edge]
    
    class Config:
        title = "Graph of Thought"
        description = "A representation of a thought process or idea flow through interconnected nodes and edges"
    nodes: list[Node]
    edges: list[Edge]

export = GraphOfThought