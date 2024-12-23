# Memos Module Generator

This module automatically generates new memos modules following a standardized interface. It uses a step-by-step approach to create all necessary components of a module, from documentation to implementation and testing.

## Features

- Automated generation of module documentation (.md files)
- Test case/evaluation generation
- Export interface definition
- Function scaffolding and implementation
- HTML template generation
- Route configuration
- Iterative refinement until completion

## Steps

1. Documentation Generation
   - Creates initial markdown documentation
   - Validates documentation structure and completeness
   
2. Test Case Generation
   - Generates comprehensive test cases
   - Creates evaluation scenarios
   - Validates test coverage

3. Export Interface
   - Defines public module interface
   - Specifies input/output contracts
   - Documents dependencies

4. Function Generation
   - Creates main function skeleton
   - Implements supporting functions
   - Validates function signatures

5. Frontend Integration
   - Generates HTML templates
   - Creates route handlers
   - Implements frontend interactions

6. Refinement Loop
   - Validates against test cases
   - Refactors code as needed
   - Continues until DONE state

## Usage

```python
from memos._10_memos_module.memos_module import main as memos_module_main

# Example: Generate a new knowledge graph module
module_spec = {
    "description": """Create a module that generates knowledge graphs from text input. 
    It should extract entities and their relationships, classify entity types, 
    and create a visual representation of the knowledge. The module should handle 
    complex relationships, support different types of entities (people, organizations, 
    concepts, etc.), and provide methods to serialize the graph structure. Include 
    capabilities for merging multiple graphs and detecting conflicting information.""",
    
    # Optional fields
    "name": "knowledge_graph_gpt4",  # If not provided, will be generated from description
    "features": [  # Optional - if not provided, will be inferred from description
        "Entity extraction",
        "Relationship mapping",
        "Graph visualization"
    ],
    "dependencies": [  # Optional - if not provided, will be inferred from description
        "openai",
        "networkx"
    ],
    "module_context": [  # Optional - documents to help generate the module implementation
        """Example knowledge graph implementation:
        class KnowledgeGraph:
            def __init__(self):
                self.nodes = []
                self.edges = []
                
            def add_node(self, node):
                self.nodes.append(node)
                
            def add_edge(self, source, target, relation):
                self.edges.append((source, target, relation))
        ...""",
        """Graph serialization example:
        def to_json(self):
            return {
                'nodes': [node.to_dict() for node in self.nodes],
                'edges': [edge.to_dict() for edge in self.edges]
            }
        """
    ],
    "test_context": [  # Optional - documents to help generate test cases
        """Test case for entity extraction:
        input_text = "OpenAI developed ChatGPT in 2022"
        expected_entities = ["OpenAI", "ChatGPT"]
        ...""",
        """Test case for relationship detection:
        text = "Einstein developed the theory of relativity"
        expected_relation = {
            'source': 'Einstein',
            'relation': 'developed',
            'target': 'theory of relativity'
        }
        ..."""
    ]
}

new_module = memos_module_main(module_spec)

print("\nGenerated Module Summary:")
print(f"Module Name: {new_module.name}")
print(f"Files Generated: {len(new_module.files)}")
for file in new_module.files:
    print(f"- {file.path}")

print("\nTest Cases Generated:")
for test in new_module.test_cases:
    print(f"- {test.name}: {test.description}")
```

## Module Structure

Generated modules follow this standard structure:
```
memos/
├── memos/
│   └── _<module_number>_<module_name>/
│       ├── __init__.py
│       ├── <module_name>.md
│       ├── <module_name>.py
│       ├── routes.py
│       └── test_<module_name>.py
├── templates/
│   └── <module_name>.html
└── app.py
```

## Installation

After generating the module:

1. Add the imports to app.py:
```python
from memos._<module_number>_<module_name>.<module_name> import main as <module_name>_main
from memos._<module_number>_<module_name>.routes import routes as <module_name>_routes
```

2. Register the routes in app.py:
```python
# Add <module_name> routes
for route in <module_name>_routes:
    app.add_url_rule(
        route["rule"],
        route["endpoint"],
        route["view_func"],
        methods=route["methods"]
    )
```

## Testing

Each generated module includes:
- Unit tests for core functionality
- Integration tests for module interfaces
- Evaluation scenarios for real-world use cases
- Performance benchmarks

## Refinement Process

The module generator implements an iterative refinement loop:
1. Generate initial implementation
2. Run test suite
3. Analyze test results
4. Refactor and improve
5. Repeat until all tests pass and quality metrics are met
