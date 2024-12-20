from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List, Dict, Optional, Union
import json
import logging
import time
import os
from datetime import datetime, timedelta
import random
from openai import RateLimitError
import networkx as nx
from pyvis.network import Network
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI()
MODEL = "gpt-4o-2024-08-06"

class MemoryUnit(BaseModel):
    """Base class for all memory units in the system"""
    id: str = Field(..., description="Unique identifier")
    content: str = Field(..., description="The actual content/text")
    created_at: datetime = Field(default_factory=datetime.now)
    last_reviewed: datetime = Field(default_factory=datetime.now)
    confidence: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Confidence level in this memory (0-1)"
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance level of this memory (0-1)"
    )
    review_interval: timedelta = Field(
        default=timedelta(days=1),
        description="Time interval for next review"
    )

class Value(MemoryUnit):
    """Represents core values and beliefs"""
    type: str = "value"
    description: str = Field(..., description="Detailed description of the value")
    
class Goal(MemoryUnit):
    """Represents goals derived from values"""
    type: str = "goal"
    parent_value_id: str = Field(..., description="ID of the parent value")
    target_date: Optional[datetime] = Field(None, description="Target completion date")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Progress towards goal (0-1)"
    )

class Action(MemoryUnit):
    """Represents concrete actions to achieve goals"""
    type: str = "action"
    parent_goal_id: str = Field(..., description="ID of the parent goal")
    status: str = Field(
        default="pending",
        description="Status of the action (pending/in_progress/completed)"
    )
    due_date: Optional[datetime] = Field(None, description="Due date for the action")

class Reflection(MemoryUnit):
    """Represents reflections on values, goals, or actions"""
    type: str = "reflection"
    reference_id: str = Field(..., description="ID of the referenced value/goal/action")
    reflection_type: str = Field(
        ..., 
        description="Type of reflection (success/challenge/learning/adjustment)"
    )

class MemoryGraph(BaseModel):
    """Represents the memory graph structure"""
    nodes: Dict[str, Dict] = Field(default_factory=dict)
    edges: List[Dict] = Field(default_factory=list)

    def add_node(self, id: str, content: str, type: str, **attrs):
        """Add a node to the graph with natural language content"""
        self.nodes[id] = {
            "id": id,
            "content": content,
            "type": type,
            **attrs
        }
    
    def add_edge(self, source: str, target: str, relation: str, **attrs):
        """Add an edge between nodes with natural language relation"""
        self.edges.append({
            "source": source,
            "target": target,
            "relation": relation,
            **attrs
        })
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for visualization/analysis"""
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, attrs in self.nodes.items():
            G.add_node(node_id, **attrs)
            
        # Add edges
        for edge in self.edges:
            G.add_edge(
                edge["source"], 
                edge["target"], 
                relation=edge["relation"],
                **{k:v for k,v in edge.items() if k not in ["source", "target", "relation"]}
            )
            
        return G

class LifeLTM(BaseModel):
    """Main class for managing the life long-term memory system"""
    memory_graph: MemoryGraph = Field(default_factory=MemoryGraph)
    
    def add_value(self, value: Value):
        """Add a value to the memory graph"""
        self.memory_graph.add_node(
            id=value.id,
            content=value.content,
            type="value",
            description=value.description,
            importance=value.importance,
            confidence=value.confidence,
            created_at=value.created_at,
            last_reviewed=value.last_reviewed,
            review_interval=value.review_interval
        )
    
    def add_goal(self, goal: Goal):
        """Add a goal and connect to parent value"""
        self.memory_graph.add_node(
            id=goal.id,
            content=goal.content,
            type="goal",
            importance=goal.importance,
            confidence=goal.confidence,
            progress=goal.progress,
            target_date=goal.target_date,
            created_at=goal.created_at,
            last_reviewed=goal.last_reviewed,
            review_interval=goal.review_interval
        )
        self.memory_graph.add_edge(
            source=goal.parent_value_id,
            target=goal.id,
            relation="supports"
        )
    
    def add_action(self, action: Action):
        """Add an action and connect to parent goal"""
        self.memory_graph.add_node(
            id=action.id,
            content=action.content,
            type="action",
            importance=action.importance,
            confidence=action.confidence,
            status=action.status,
            due_date=action.due_date,
            created_at=action.created_at,
            last_reviewed=action.last_reviewed,
            review_interval=action.review_interval
        )
        self.memory_graph.add_edge(
            source=action.parent_goal_id,
            target=action.id,
            relation="achieves"
        )
    
    def add_reflection(self, reflection: Reflection):
        """Add a reflection and connect to referenced memory"""
        self.memory_graph.add_node(
            id=reflection.id,
            content=reflection.content,
            type="reflection",
            reflection_type=reflection.reflection_type,
            importance=reflection.importance,
            confidence=reflection.confidence,
            created_at=reflection.created_at,
            last_reviewed=reflection.last_reviewed,
            review_interval=reflection.review_interval
        )
        self.memory_graph.add_edge(
            source=reflection.id,
            target=reflection.reference_id,
            relation="reflects_on"
        )

    def get_value_chain(self, action_id: str) -> Dict[str, str]:
        """Trace an action back to its supporting goal and value"""
        action = next((a for a in self.actions if a.id == action_id), None)
        if not action:
            return {}
            
        goal = next((g for g in self.goals if g.id == action.parent_goal_id), None)
        if not goal:
            return {'action': action.content}
            
        value = next((v for v in self.values if v.id == goal.parent_value_id), None)
        if not value:
            return {'action': action.content, 'goal': goal.content}
            
        return {
            'value': value.content,
            'goal': goal.content,
            'action': action.content
        }
    
    def get_related_reflections(self, memory_id: str) -> List[Reflection]:
        """Get all reflections related to a specific memory unit"""
        return [r for r in self.reflections if r.reference_id == memory_id]

def handle_rate_limit(func):
    """Decorator to handle rate limit errors with exponential backoff"""
    def wrapper(*args, **kwargs):
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                logger.warning(f"Rate limit hit. Waiting {delay:.2f}s before retry")
                time.sleep(delay)
        
    return wrapper

class AlignmentSuggestion(BaseModel):
    """Single alignment suggestion"""
    message: str = Field(..., description="Suggestion message")

class ValueAlignment(BaseModel):
    """Response format for value alignment suggestions"""
    suggestions: List[AlignmentSuggestion] = Field(
        ..., 
        description="List of suggested improvements"
    )

@handle_rate_limit
def analyze_alignment(ltm: LifeLTM) -> List[str]:
    """Analyze alignment between values, goals, and actions"""
    # Convert LTM data to a format suitable for the API
    ltm_data = ltm.model_dump_json()
    
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": """Analyze the alignment between values, goals, and actions.
                Look for:
                1. Goals that don't clearly support any values
                2. Actions that don't clearly support any goals
                3. Values without supporting goals
                4. Goals without concrete actions
                
                Return suggestions in the format:
                {
                    "suggestions": [
                        {"message": "Add specific goals for X value"},
                        {"message": "Break down Y goal into smaller actions"}
                    ]
                }"""},
                {"role": "user", "content": ltm_data}
            ],
            response_format=ValueAlignment,
            temperature=0
        )
        
        # Extract just the message strings from the suggestions
        return [suggestion.message for suggestion in completion.choices[0].message.parsed.suggestions]
    except Exception as e:
        logger.error(f"Alignment analysis failed: {str(e)}")
        return ["Unable to analyze alignment at this time"]

def parse_timedelta(td_str: str) -> timedelta:
    """Parse a timedelta string into a timedelta object"""
    if isinstance(td_str, timedelta):
        return td_str
        
    # Handle "X days, HH:MM:SS" format
    if "day" in td_str:
        days = int(td_str.split()[0])
        return timedelta(days=days)
    
    # Handle "HH:MM:SS" format
    try:
        hours, minutes, seconds = map(int, td_str.split(':'))
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except:
        # Default to 1 day if parsing fails
        return timedelta(days=1)

def get_due_for_review(ltm: LifeLTM) -> List[MemoryUnit]:
    """Get all memory units due for review"""
    now = datetime.now()
    due_items = []
    
    # Get all nodes from the memory graph
    for node_id, node_data in ltm.memory_graph.nodes.items():
        # Convert node data back to MemoryUnit
        if 'last_reviewed' not in node_data or 'review_interval' not in node_data:
            continue
            
        last_reviewed = datetime.fromisoformat(str(node_data['last_reviewed']))
        review_interval = parse_timedelta(str(node_data['review_interval']))
        
        next_review = last_reviewed + review_interval
        if now >= next_review:
            # Create appropriate memory unit based on type
            if node_data['type'] == 'value':
                memory = Value(
                    id=node_id,
                    content=node_data['content'],
                    description=node_data.get('description', ''),
                    importance=node_data.get('importance', 0.5),
                    confidence=node_data.get('confidence', 0.5),
                    last_reviewed=last_reviewed,
                    review_interval=review_interval
                )
            elif node_data['type'] == 'goal':
                memory = Goal(
                    id=node_id,
                    content=node_data['content'],
                    parent_value_id=node_data.get('parent_value_id', ''),
                    importance=node_data.get('importance', 0.5),
                    confidence=node_data.get('confidence', 0.5),
                    progress=node_data.get('progress', 0.0),
                    last_reviewed=last_reviewed,
                    review_interval=review_interval
                )
            elif node_data['type'] == 'action':
                memory = Action(
                    id=node_id,
                    content=node_data['content'],
                    parent_goal_id=node_data.get('parent_goal_id', ''),
                    importance=node_data.get('importance', 0.5),
                    confidence=node_data.get('confidence', 0.5),
                    status=node_data.get('status', 'pending'),
                    last_reviewed=last_reviewed,
                    review_interval=review_interval
                )
            elif node_data['type'] == 'reflection':
                memory = Reflection(
                    id=node_id,
                    content=node_data['content'],
                    reference_id=node_data.get('reference_id', ''),
                    reflection_type=node_data.get('reflection_type', 'learning'),
                    importance=node_data.get('importance', 0.5),
                    confidence=node_data.get('confidence', 0.5),
                    last_reviewed=last_reviewed,
                    review_interval=review_interval
                )
            else:
                continue
                
            due_items.append(memory)
    
    # Sort by importance and overdue amount
    due_items.sort(
        key=lambda x: (
            x.importance,
            (now - (x.last_reviewed + x.review_interval)).total_seconds()
        ),
        reverse=True
    )
    
    return due_items

def visualize_memory_graph(ltm: LifeLTM, output_path: str = None):
    """Create an interactive visualization of the memory graph"""
    G = ltm.memory_graph.to_networkx()
    
    # Create Pyvis network
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black"
    )
    
    # Configure physics
    net.force_atlas_2based()
    
    # Color scheme for different node types
    colors = {
        "value": "#ff7675",    # Red
        "goal": "#74b9ff",     # Blue
        "action": "#55efc4",   # Green
        "reflection": "#ffeaa7" # Yellow
    }
    
    # Add nodes
    for node_id, attrs in G.nodes(data=True):
        net.add_node(
            node_id,
            label=attrs["content"],
            title=f"Type: {attrs['type']}\n{attrs.get('description', '')}",
            color=colors[attrs["type"]]
        )
    
    # Add edges
    for source, target, attrs in G.edges(data=True):
        net.add_edge(
            source,
            target,
            title=attrs["relation"],
            arrows="to"
        )
    
    # Save visualization
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "visualizations",
            f"memory_graph_{time.strftime('%Y%m%d_%H%M%S')}.html"
        )
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    net.save_graph(output_path)
    logger.info(f"Saved visualization to {output_path}")
    return output_path

def save_ltm(ltm: LifeLTM, filename: str = None):
    """Save LTM to JSON file preserving graph structure"""
    try:
        ltm_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(ltm_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"life_ltm_{timestamp}.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        full_path = os.path.join(data_dir, filename)
        
        # Save as JSON with graph structure
        with open(full_path, 'w') as f:
            json.dump(
                ltm.memory_graph.model_dump(),
                f,
                indent=2,
                default=str
            )
        
        logger.info(f"Saved LTM to {full_path}")
        
        # Also create visualization
        vis_path = visualize_memory_graph(ltm)
        
        return filename
    except Exception as e:
        raise RuntimeError(f"Failed to save LTM: {str(e)}")

def load_ltm(filename: str) -> LifeLTM:
    """Load LTM from JSON file"""
    try:
        ltm_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(ltm_dir, 'data')
        full_path = os.path.join(data_dir, filename)
        
        with open(full_path, 'r') as f:
            data = json.load(f)
            
        # Convert string dates back to datetime
        for item_list in [data['values'], data['goals'], data['actions'], data['reflections']]:
            for item in item_list:
                item['created_at'] = datetime.fromisoformat(item['created_at'])
                item['last_reviewed'] = datetime.fromisoformat(item['last_reviewed'])
                item['review_interval'] = timedelta(
                    seconds=eval(item['review_interval'].replace('days', '* 86400'))
                )
                if 'target_date' in item and item['target_date']:
                    item['target_date'] = datetime.fromisoformat(item['target_date'])
                if 'due_date' in item and item['due_date']:
                    item['due_date'] = datetime.fromisoformat(item['due_date'])
        
        return LifeLTM(**data)
    except Exception as e:
        raise RuntimeError(f"Failed to load LTM: {str(e)}")

def main():
    """Example usage of the Life LTM system"""
    # Create a new LTM
    ltm = LifeLTM()
    
    # Add a value
    value = Value(
        id="v1",
        content="Continuous Learning",
        description="Commitment to lifelong learning and growth",
        importance=0.9
    )
    ltm.add_value(value)
    
    # Add a goal
    goal = Goal(
        id="g1",
        content="Master Python Programming",
        parent_value_id="v1",
        importance=0.8,
        target_date=datetime.now() + timedelta(days=90)
    )
    ltm.add_goal(goal)
    
    # Add an action
    action = Action(
        id="a1",
        content="Complete Python course on Coursera",
        parent_goal_id="g1",
        importance=0.7,
        due_date=datetime.now() + timedelta(days=30)
    )
    ltm.add_action(action)
    
    # Save the LTM
    filename = save_ltm(ltm)
    print(f"\nSaved LTM to {filename}")
    
    # Analyze alignment
    suggestions = analyze_alignment(ltm)
    print("\nAlignment Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")
    
    # Get items due for review
    due_items = get_due_for_review(ltm)
    print("\nItems Due for Review:")
    for item in due_items:
        print(f"- {item.content} (Last reviewed: {item.last_reviewed})")
    
    # Create visualization
    vis_path = visualize_memory_graph(ltm)
    print(f"\nVisualization saved to {vis_path}")
    
    # Get reasoning trace for a decision
    print("\nGetting reasoning trace for learning decision...")
    trace = get_reasoning_trace(
        ltm, 
        "Should I spend more time on Python exercises or move on to a new programming language?"
    )
    
    print("\nReasoning Trace:")
    print("\nRelevant Context:")
    for context in trace.context:
        print(f"- {context}")
    
    print("\nReasoning Process:")
    print(trace.reasoning)
    
    print("\nConclusion:")
    print(trace.conclusion)

class SampleLTM:
    """Provides sample data for testing and demonstration"""
    
    @staticmethod
    def create_learning_sample() -> LifeLTM:
        """Creates a sample LTM focused on learning goals"""
        ltm = LifeLTM()
        
        # Add value
        value = Value(
            id="v1",
            content="Continuous Learning",
            description="Commitment to lifelong learning and growth",
            importance=0.9,
            confidence=0.8
        )
        ltm.add_value(value)
        
        # Add goal
        goal = Goal(
            id="g1",
            content="Master Python Programming",
            parent_value_id="v1",
            importance=0.8,
            confidence=0.7,
            target_date=datetime.now() + timedelta(days=90),
            progress=0.3
        )
        ltm.add_goal(goal)
        
        # Add action
        action = Action(
            id="a1",
            content="Complete Python course on Coursera",
            parent_goal_id="g1",
            importance=0.7,
            confidence=0.6,
            status="in_progress",
            due_date=datetime.now() + timedelta(days=30)
        )
        ltm.add_action(action)
        
        # Add reflection
        reflection = Reflection(
            id="r1",
            content="Making good progress on Python basics",
            reference_id="g1",
            importance=0.6,
            confidence=0.7,
            reflection_type="learning"
        )
        ltm.add_reflection(reflection)
        
        return ltm
    
    @staticmethod
    def get_test_cases() -> List[Dict]:
        """Returns test cases from test_data.jsonl"""
        test_cases = []
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'test_data.jsonl'
        )
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))
        
        return test_cases

# Add new class for text storage
class TextMemoryStore:
    """Stores memories in natural language text format"""
    
    @staticmethod
    def memory_to_text(memory: MemoryUnit) -> str:
        """Convert a memory unit to natural language text"""
        if isinstance(memory, Value):
            return f"""Value: {memory.content}
Description: {memory.description}
Importance: {memory.importance:.1f}
Confidence: {memory.confidence:.1f}
Last Reviewed: {memory.last_reviewed.strftime('%Y-%m-%d')}
---"""
        
        elif isinstance(memory, Goal):
            return f"""Goal: {memory.content}
Supports Value: {memory.parent_value_id}
Progress: {memory.progress*100:.0f}%
Target Date: {memory.target_date.strftime('%Y-%m-%d') if memory.target_date else 'Ongoing'}
Importance: {memory.importance:.1f}
---"""
        
        elif isinstance(memory, Action):
            return f"""Action: {memory.content}
Supports Goal: {memory.parent_goal_id}
Status: {memory.status}
Due Date: {memory.due_date.strftime('%Y-%m-%d') if memory.due_date else 'No due date'}
---"""
        
        elif isinstance(memory, Reflection):
            return f"""Reflection ({memory.reflection_type}): {memory.content}
References: {memory.reference_id}
Date: {memory.created_at.strftime('%Y-%m-%d')}
---"""
    
    @staticmethod
    def save_memories(ltm: LifeLTM, filename: str):
        """Save all memories as natural language text"""
        ltm_dir = os.path.dirname(os.path.abspath(__file__))
        text_dir = os.path.join(ltm_dir, 'text_memories')
        os.makedirs(text_dir, exist_ok=True)
        
        full_path = os.path.join(text_dir, filename)
        
        with open(full_path, 'w') as f:
            # Write values
            f.write("=== VALUES ===\n\n")
            for value in ltm.values:
                f.write(TextMemoryStore.memory_to_text(value) + "\n")
                
            # Write goals
            f.write("\n=== GOALS ===\n\n")
            for goal in ltm.goals:
                f.write(TextMemoryStore.memory_to_text(goal) + "\n")
                
            # Write actions
            f.write("\n=== ACTIONS ===\n\n")
            for action in ltm.actions:
                f.write(TextMemoryStore.memory_to_text(action) + "\n")
                
            # Write reflections
            f.write("\n=== REFLECTIONS ===\n\n")
            for reflection in ltm.reflections:
                f.write(TextMemoryStore.memory_to_text(reflection) + "\n")

class ReasoningTrace(BaseModel):
    """Represents a reasoning trace for decisions and suggestions"""
    context: List[str] = Field(..., description="Relevant memory units used for reasoning")
    reasoning: str = Field(..., description="Step-by-step reasoning process")
    conclusion: str = Field(..., description="Final conclusion or suggestion")

@handle_rate_limit
def get_reasoning_trace(ltm: LifeLTM, query: str) -> ReasoningTrace:
    """Get reasoning trace for a specific query using relevant memories"""
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": """Provide reasoning traces based on the person's knowledge base.
            Show:
            1. Which memories were relevant to the decision
            2. Step-by-step reasoning process
            3. Clear conclusion that ties back to values"""},
            {"role": "user", "content": f"Query: {query}\nMemories: {ltm.model_dump_json()}"}
        ],
        response_format=ReasoningTrace,
        temperature=0
    )
    return completion.choices[0].message.parsed

if __name__ == "__main__":
    main()

export = {
    "default": main,
    "LifeLTM": LifeLTM,
    "save_ltm": save_ltm,
    "load_ltm": load_ltm,
    "analyze_alignment": analyze_alignment,
    "get_due_for_review": get_due_for_review,
    "SampleLTM": SampleLTM,
    "visualize_memory_graph": visualize_memory_graph
}
