# Knowledge Graph Generator

This module provides functionality to generate knowledge graphs from text input using GPT-4o. It extracts entities, their relationships, and creates a structured representation of the knowledge contained in the text.

## Features

Using GPT-4o:
- Entity extraction with type classification
- Relationship extraction between entities
- Semantic clustering of similar entities
- JSON serialization of knowledge graphs
- Automatic filename generation based on graph contents

## Usage

```python
from memos._7_knowledge_graph_4o.knowledge_graph_4o import main as knowledge_graph_4o_main

# Example usage
text = """
OpenAI was founded by Sam Altman and others in San Francisco. 
The company developed ChatGPT, which uses large language models for natural language processing.
Microsoft invested heavily in OpenAI and integrated ChatGPT into their Bing search engine.
"""

kg = knowledge_graph_4o_main(text)

print("\nKnowledge Graph Summary:")
print(f"Nodes: {len(kg.nodes)}")
for node in kg.nodes:
    print(f"- {node.label} ({node.type})")

print(f"\nEdges: {len(kg.edges)}")
for edge in kg.edges:
    print(f"- {edge.source} --[{edge.relation}]--> {edge.target}")
```



