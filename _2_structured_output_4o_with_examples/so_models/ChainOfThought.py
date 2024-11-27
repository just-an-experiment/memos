# Represents a complete chain of thought with interconnected nodes and edges.

from pydantic import BaseModel, Field

class Edge(BaseModel):
    """Represents an edge in a chain of thought, connecting two nodes."""
    source_node_id: int = Field(..., description="The ID of the source node of the edge.")
    target_node_id: int = Field(..., description="The ID of the target node of the edge.")
    description: str = Field(..., description="A description or label for the edge, explaining the relationship between the source and target nodes.")

class Node(BaseModel):
    """Represents a node in a chain of thought, containing the thought content."""
    node_id: int = Field(..., description="The unique identifier for the node.")
    content: str = Field(..., description="The thought content or information encapsulated within the node.")
    description: str = Field(None, description="An optional description of the node, elaborating its significance.")

class ChainOfThought(BaseModel):
    """Represents a complete chain of thought with interconnected nodes and edges."""
    nodes: list[Node] = Field(..., description="A list of nodes forming part of the chain of thought.")
    edges: list[Edge] = Field(..., description="A list of edges representing connections between the nodes.")

# Example data that matches the model schema
examples = [
    {'nodes': [{'node_id': 1, 'content': 'Waking up early in the morning', 'description': 'The initial step that sets the tone for the day.'}, {'node_id': 2, 'content': 'Having a healthy breakfast', 'description': 'Important to provide energy for the morning activities.'}, {'node_id': 3, 'content': 'Engaging in physical exercise', 'description': 'Contributes to physical well-being and alertness.'}, {'node_id': 4, 'content': "Planning the day's agenda", 'description': 'Ensures a structured and efficient day.'}], 'edges': [{'source_node_id': 1, 'target_node_id': 2, 'description': 'Waking up early allows time for breakfast.'}, {'source_node_id': 2, 'target_node_id': 3, 'description': 'A nutritious breakfast fuels physical exercise.'}, {'source_node_id': 3, 'target_node_id': 4, 'description': 'Exercise boosts mental clarity for planning the day.'}]},
    {'nodes': [{'node_id': 1, 'content': 'Identifying a market gap', 'description': 'The foundational step in product development.'}, {'node_id': 2, 'content': 'Conducting customer surveys', 'description': 'Gathers insights and validates assumptions.'}, {'node_id': 3, 'content': 'Developing a product prototype', 'description': 'Initial version to test functionality.'}, {'node_id': 4, 'content': 'Collecting feedback and refining', 'description': 'Improving the product based on user feedback.'}], 'edges': [{'source_node_id': 1, 'target_node_id': 2, 'description': 'Surveys further understand the identified gap.'}, {'source_node_id': 2, 'target_node_id': 3, 'description': 'Survey results inform prototype development.'}, {'source_node_id': 3, 'target_node_id': 4, 'description': 'Feedback on the prototype leads to refinements.'}]},
    {'nodes': [{'node_id': 1, 'content': 'Choosing a vacation destination', 'description': 'The first step in planning a trip.'}, {'node_id': 2, 'content': 'Researching accommodation options', 'description': 'Finding suitable places to stay.'}, {'node_id': 3, 'content': 'Booking flights', 'description': 'Securing travel arrangements.'}, {'node_id': 4, 'content': 'Creating a travel itinerary', 'description': 'Planning the daily activities.'}], 'edges': [{'source_node_id': 1, 'target_node_id': 2, 'description': 'The destination influences accommodation choices.'}, {'source_node_id': 2, 'target_node_id': 3, 'description': 'Knowing accommodations aids in booking flights.'}, {'source_node_id': 3, 'target_node_id': 4, 'description': 'Flight dates help structure the itinerary.'}]},
    {'nodes': [{'node_id': 1, 'content': 'Setting personal fitness goals', 'description': 'The initial step for shaping fitness routines.'}, {'node_id': 2, 'content': 'Choosing a workout plan', 'description': 'Aligns with the set goals effectively.'}, {'node_id': 3, 'content': 'Tracking daily progress', 'description': 'Monitors adherence and results.'}, {'node_id': 4, 'content': 'Adjusting the workout plan accordingly', 'description': 'Ensures continued progress and effectiveness.'}], 'edges': [{'source_node_id': 1, 'target_node_id': 2, 'description': 'Goals dictate the type of workout plan.'}, {'source_node_id': 2, 'target_node_id': 3, 'description': 'The plan provides a structure for tracking progress.'}, {'source_node_id': 3, 'target_node_id': 4, 'description': 'Progress tracking informs necessary adjustments.'}]},
    {'nodes': [{'node_id': 1, 'content': 'Brainstorming new business ideas', 'description': 'Generates a pool of innovative ideas.'}, {'node_id': 2, 'content': 'Conducting market analysis', 'description': 'Evaluates the viability of selected ideas.'}, {'node_id': 3, 'content': 'Developing a business model', 'description': 'Outlines the strategic approach for the business.'}, {'node_id': 4, 'content': 'Pitching to potential investors', 'description': 'Secures the necessary capital for startup.'}], 'edges': [{'source_node_id': 1, 'target_node_id': 2, 'description': 'Market analysis narrows down the brainstormed ideas.'}, {'source_node_id': 2, 'target_node_id': 3, 'description': 'Analysis results guide the business model development.'}, {'source_node_id': 3, 'target_node_id': 4, 'description': 'The business model forms the basis of the pitch.'}]},
]


export = {
    'default': ChainOfThought,
    'examples': examples
}