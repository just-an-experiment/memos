# Represents a graph structure of interconnected chains of thought.

from pydantic import BaseModel, Field

class ChainOfThought(BaseModel):
    """
    Represents a step or link in a chain of thought within a graph.
    """
    id: int = Field(..., description="Unique identifier for the chain of thought.")
    content: str = Field(..., description="The content or description of the thought.")
    connections: list[int] = Field(..., description="List of IDs representing connections to other chains of thought.")
    

class GraphOfThought(BaseModel):
    """
    Represents a graph structure of interconnected chains of thought.
    """
    chains_of_thought: list[ChainOfThought] = Field(..., description="List of all chains of thought in the graph.")
    title: str = Field(..., description="Title of the graph of thought.")
    description: str = Field(None, description="Optional detailed description of the graph.")

# Example data that matches the model schema
examples = [
    {'chains_of_thought': [{'id': 1, 'content': 'Identify the problem domain.', 'connections': [2]}, {'id': 2, 'content': 'Gather relevant resources and information.', 'connections': [3, 4]}, {'id': 3, 'content': 'Formulate a hypothesis or potential solution.', 'connections': [5]}, {'id': 4, 'content': 'Consult with domain experts.', 'connections': [5, 6]}, {'id': 5, 'content': 'Develop a structured plan or proposal.', 'connections': [7]}, {'id': 6, 'content': 'Review similar past solutions.', 'connections': [7]}, {'id': 7, 'content': 'Execute the plan and monitor progress.', 'connections': [8, 9]}, {'id': 8, 'content': 'Analyze results and evaluate outcomes.', 'connections': [10]}, {'id': 9, 'content': 'Apply feedback for iterative improvement.', 'connections': [10]}, {'id': 10, 'content': 'Reflect on the process and document insights.', 'connections': []}], 'title': 'Problem-Solving Process', 'description': 'A detailed look into the process of solving complex problems by logically structuring thoughts and making informed decisions.'},
]


export = {
    'default': GraphOfThought,
    'examples': examples
}