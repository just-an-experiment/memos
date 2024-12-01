from typing import Dict, Any, Callable, List, Tuple
import heapq

def a_star_algorithm(graph: Dict[Any, Dict[Any, float]], start: Any, end: Any, heuristic: Callable[[Any, Any], float]) -> List[Any]:
    """
    Implements the A* algorithm to find the shortest path between a start node and an end node in a weighted graph.
    
    :param graph: A dictionary representing the graph, where each key is a node and its value is a dictionary of adjacent nodes with their corresponding weights.
    :param start: The start node for the pathfinding.
    :param end: The end node for the pathfinding.
    :param heuristic: A function that estimates the cost from a node to the end node.
    :return: A list representing the shortest path from the start node to the end node, including both.
    """
    open_set = [(0, start)]  # Priority queue of nodes to explore, initialized with the start node
    came_from: Dict[Any, Any] = {}  # To store the path traversed, mapping current node to its predecessor
    g_score: Dict[Any, float] = {start: 0}  # Cost of the path from start to a node
    f_score: Dict[Any, float] = {start: heuristic(start, end)}  # Estimated total cost from start to end passing through a node
    
    while open_set:
        # Get the node with the lowest f_score from the open set
        current_f, current = heapq.heappop(open_set)
        
        # If the current node is the end node, reconstruct the path and return it
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return the path from start to end
        
        # Explore neighbors of the current node
        for neighbor, weight in graph.get(current, {}).items():
            tentative_g_score = g_score.get(current, float('inf')) + weight
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                # This path to neighbor is better than any previous one, so store it
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # If no path is found to the end, return an empty list
    return []

def test_a_star_algorithm():
    assert a_star_algorithm({'A': {'B': 1, 'C': 3}, 'B': {'C': 1, 'D': 1}, 'C': {'D': 1}, 'D': {}}, 'A', 'D', lambda x, y: 0) == ['A', 'B', 'D']
    assert a_star_algorithm({'A': {'B': 2, 'C': 2}, 'B': {'D': 1}, 'C': {'D': 2}, 'D': {}}, 'A', 'D', lambda x, y: 0) == ['A', 'B', 'D']
    assert a_star_algorithm({'A': {'B': 1, 'C': 1}, 'B': {'D': 1}, 'C': {'B': 1, 'D': 1}, 'D': {}}, 'A', 'D', lambda x, y: 0) == ['A', 'C', 'D']
    assert a_star_algorithm({'A': {'B': 2, 'C': 2}, 'B': {'C': -1, 'D': 4}, 'C': {'D': 1}, 'D': {}}, 'A', 'D', lambda x, y: 0) == ['A', 'C', 'D']
    assert a_star_algorithm({'A': {'B': 1}, 'B': {}, 'C': {'D': 1}, 'D': {}}, 'A', 'D', lambda x, y: 0) == []
    assert a_star_algorithm({'A': {'B': 1}, 'B': {'A': 1}, 'C': {'D': 1}, 'D': {'C': 1}}, 'A', 'C', lambda x, y: 0) == []
    assert a_star_algorithm({'A': {'B': 1, 'C': 5}, 'B': {'C': 1, 'D': 4}, 'C': {'D': 1}, 'D': {}}, 'A', 'D', lambda x, y: {'A': 5, 'B': 3, 'C': 1, 'D': 0}[x]) == ['A', 'B', 'C', 'D']
    assert a_star_algorithm({'A': {'B': 3}, 'B': {'C': 1}, 'C': {'D': 1, 'E': 5}, 'D': {'E': 2}, 'E': {}}, 'A', 'E', lambda x, y: {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0}[x]) == ['A', 'B', 'C', 'D', 'E']

export = {
    'tests': test_a_star_algorithm,
    'default': a_star_algorithm
}