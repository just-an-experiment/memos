# Life Long-Term Memory (LTM) System

A hierarchical memory system that helps people align their values, goals, and actions while integrating with LLM-powered assistance.

## Overview

The Life LTM system provides:

1. A structured way to map personal values → goals → actions
2. Natural language storage of memories in a graph structure
3. Spaced repetition for reviewing important items
4. Reasoning traces for decisions
5. Visual representation of the memory network
6. AI-assisted alignment analysis

## Core Principles

- "Plain text is preferred" - All memories stored in natural language
- "Do one thing and do it well" - Each component has a focused purpose
- "Everything is a file" - Memories stored in accessible text/JSON formats
- "Script shell automation" - Automatable through clear interfaces
- "Silence is golden" - Only meaningful information displayed

## Memory Types

### Values
Core beliefs and principles that guide decision making

```python
Value(
content="Continuous Learning",
description="Commitment to lifelong learning and growth",
importance=0.9
)
```

### Goals
Specific objectives derived from values

```python
Goal(
content="Master Python Programming",
parent_value_id="v1",
progress=0.3,
target_date="2024-06-10"
)
```


### Actions
Concrete steps to achieve goals
python

```python
Action(
content="Complete Python course on Coursera",
parent_goal_id="g1",
status="in_progress",
due_date="2024-04-10"
)
```


### Reflections
Thoughts and learnings about values, goals, or actions
```python
Reflection(
content="Making good progress on Python basics",
parent_goal_id="g1",
reflection_type="learning"
)
```

## Key Features

### Memory Graph
- Stores memories as nodes in a directed graph
- Edges represent relationships (supports, achieves, reflects_on)
- Natural language content in nodes and edges
- Interactive visualization with color coding:
  - Values (Red)
  - Goals (Blue)
  - Actions (Green)
  - Reflections (Yellow)

### Spaced Repetition
- Automatically schedules reviews based on:
  - Importance
  - Confidence
  - Review history
  - Memory type
- FSRS-inspired algorithm adjusts intervals

### Reasoning Traces
- Provides explainable decision paths
- Shows relevant memories used
- Step-by-step reasoning process
- Clear conclusions tied to values

### Value Alignment
- Analyzes coherence between values, goals, and actions
- Identifies gaps and misalignments
- Suggests improvements
- Maintains user control over priorities

## Usage

### Basic Example

```python
from life_ltm import LifeLTM, Value, Goal, Action
# Create new LTM
ltm = LifeLTM()
# Add value
value = Value(
id="v1",
content="Continuous Learning",
description="Commitment to lifelong learning"
)
ltm.add_value(value)
# Add goal
goal = Goal(
id="g1",
content="Master Python",
parent_value_id="v1"
)
ltm.add_goal(goal)
# Save and visualize
save_ltm(ltm)
visualize_memory_graph(ltm)
```


### Review Process
1. Get items due for review:
```python
items = ltm.get_items_due_for_review()
```


2. Update after review:
```python
new_interval = calculate_next_review(memory, review_result)
memory.review_interval = new_interval
```


### Alignment Analysis

```python
suggestions = analyze_alignment(ltm)
for suggestion in suggestions:
print(suggestion)
```

### Get Reasoning Trace

```python
trace = get_reasoning_trace(ltm, "Should I take this Python course?")
print(f"Context: {trace.context}")
print(f"Reasoning: {trace.reasoning}")
print(f"Conclusion: {trace.conclusion}")
```

## File Structure

```
life_ltm/
├── data/ # Stored memory graphs
├── text_memories/ # Natural language exports
├── visualizations/ # Graph visualizations
└── test_data.jsonl # Test cases
```


## Design Philosophy

The Life LTM system follows these key principles:

1. **User Control**: The person maintains their own value hierarchy and priorities

2. **Natural Language**: Everything stored in human-readable text

3. **Traceable Reasoning**: All decisions can be traced back to values

4. **Progressive AI Integration**: AI suggestions only after clear self-alignment

5. **Memory Maintenance**: Systematic review of important items

6. **Flexible Growth**: System grows and adapts with the user

## Future Directions

1. Enhanced pattern recognition across memories
2. Collaborative memory sharing (opt-in)
3. Integration with external knowledge bases
4. Improved visualization options
5. Mobile-friendly interfaces
6. Extended reasoning capabilities

## References

- Evergreen notes system
- FSRS spaced repetition
- Pattern languages
- Unix philosophy