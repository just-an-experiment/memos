/// Calculates the shortest path from a source node to all other nodes in a weighted graph.
/// 
/// # Arguments
/// * `graph` - A reference to a vector of vectors, where each inner vector contains tuples that 
///             represent edges (in the form of (neighbor, weight)) for each node.
/// * `source` - The index of the source node from which to calculate the shortest paths.
/// 
/// # Returns
/// A vector where the ith element represents the shortest distance from the source node to the ith node.
fn dijkstra(graph: &Vec<Vec<(usize, u32)>>, source: usize) -> Vec<u32> {
    let num_nodes = graph.len();
    let mut distances = vec![u32::MAX; num_nodes];
    let mut visited = vec![false; num_nodes];
    
    distances[source] = 0;

    for _ in 0..num_nodes {
        let mut min_distance = u32::MAX;
        let mut min_index = num_nodes;

        for i in 0..num_nodes {
            if !visited[i] && distances[i] < min_distance {
                min_distance = distances[i];
                min_index = i;
            }
        }

        if min_index == num_nodes {
            break;
        }

        visited[min_index] = true;

        for &(neighbor, weight) in &graph[min_index] {
            if distances[min_index] + weight < distances[neighbor] {
                distances[neighbor] = distances[min_index] + weight;
            }
        }
    }

    distances
}

fn test_dijkstra() {
    // Test 1: Simple case with two nodes
    let graph = vec![vec![(1, 1)], vec![]];
    let source = 0;
    let expected_output = vec![0, 1];
    assert_eq!(dijkstra(&graph, source), expected_output);
    
    // Test 2: Bidirectional connection
    let graph = vec![vec![(1, 1)], vec![(0, 1)]];
    let source = 0;
    let expected_output = vec![0, 1];
    assert_eq!(dijkstra(&graph, source), expected_output);
    
    // Test 3: Three nodes in a single path
    let graph = vec![vec![(1, 2)], vec![(2, 3)], vec![]];
    let source = 0;
    let expected_output = vec![0, 2, 5];
    assert_eq!(dijkstra(&graph, source), expected_output);
    
    // Test 4: Graph with unconnected nodes
    let graph = vec![vec![], vec![], vec![], vec![]];
    let source = 0;
    let expected_output = vec![0, u32::MAX, u32::MAX, u32::MAX];
    assert_eq!(dijkstra(&graph, source), expected_output);
    
    // Test 5: Complex graph with multiple paths
    let graph = vec![vec![(1, 1), (3, 4)], vec![(2, 2)], vec![(3, 1)], vec![]];
    let source = 1;
    let expected_output = vec![u32::MAX, 0, 2, 3];
    assert_eq!(dijkstra(&graph, source), expected_output);
}



pub use test_dijkstra as tests;
pub use dijkstra as default;