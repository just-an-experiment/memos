from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List, Dict, Set, Tuple, Optional
import json
import logging
import time
import random
from openai import RateLimitError
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI()
MODEL = "gpt-4o-2024-08-06"

current_folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))


class KnowledgeGraphNode(BaseModel):
    """Represents a node in the knowledge graph"""
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Display label for the node")
    type: str = Field(..., description="Type/category of the node")
    cluster: Optional[str] = Field(None, description="Cluster this node belongs to")

class KnowledgeGraphEdge(BaseModel):
    """Represents an edge in the knowledge graph"""
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    relation: str = Field(..., description="Type of relationship between nodes")
    
class KnowledgeGraph(BaseModel):
    """Represents the complete knowledge graph"""
    nodes: List[KnowledgeGraphNode] = Field(default_factory=list)
    edges: List[KnowledgeGraphEdge] = Field(default_factory=list)

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

class Entity(BaseModel):
    """Represents a single entity and its type"""
    name: str = Field(..., description="The entity text")
    type: str = Field(..., description="Entity type (concept, person, organization, location, event, other)")

class EntityExtraction(BaseModel):
    """Response format for entity extraction"""
    entities: List[Entity] = Field(
        ..., 
        description="List of entities and their types"
    )

class Relation(BaseModel):
    """Represents a single relation between entities"""
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Predicate describing the relationship")
    object: str = Field(..., description="Object entity")

class RelationExtraction(BaseModel):
    """Response format for relation extraction"""
    relations: List[Relation] = Field(
        ...,
        description="List of relations between entities"
    )

class ClusterResponse(BaseModel):
    """Response format for semantic clustering"""
    cluster: List[str] = Field(..., description="List of semantically similar entities")

@handle_rate_limit
def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract entities and their types from text"""
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": """Extract key entities and their types from the text.
            For each entity, identify its type as one of: concept, person, organization, location, event, other.
            Focus on important entities that would be useful in a knowledge graph."""},
            {"role": "user", "content": text}
        ],
        response_format=EntityExtraction,
        temperature=0
    )
    # Convert the structured output back to tuples
    return [(entity.name, entity.type) for entity in completion.choices[0].message.parsed.entities]

@handle_rate_limit
def extract_relations(text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """Extract relationships between entities"""
    entity_list = [e[0] for e in entities]
    entity_str = ", ".join(entity_list)
    
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"""Extract relationships between entities as (subject, predicate, object) triples.
            Valid entities are: {entity_str}
            Use only these entities as subjects and objects.
            Keep predicates concise (1-3 words) and consistent."""},
            {"role": "user", "content": text}
        ],
        response_format=RelationExtraction,
        temperature=0
    )
    # Convert the structured output back to tuples
    return [(rel.subject, rel.predicate, rel.object) 
            for rel in completion.choices[0].message.parsed.relations]

@handle_rate_limit
def cluster_similar_entities(entities: List[Tuple[str, str]]) -> Dict[str, str]:
    """Group similar entities into clusters using semantic similarity"""
    clusters = {}
    entities_by_type = {}
    
    # First group by type
    for entity, type_ in entities:
        if type_ not in entities_by_type:
            entities_by_type[type_] = []
        entities_by_type[type_].append(entity)
    
    # Then find semantic clusters within each type
    for type_, type_entities in entities_by_type.items():
        if len(type_entities) < 2:
            cluster_id = f"cluster_{type_}_{len(clusters)}"
            clusters[type_entities[0]] = cluster_id
            continue
            
        # Use GPT to find semantic clusters
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": """Find semantically similar entities that should be grouped together.
                Consider variations in:
                - tenses
                - plural forms
                - stem forms
                - capitalization
                - semantic meaning"""},
                {"role": "user", "content": f"Entities: {', '.join(type_entities)}"}
            ],
            response_format=ClusterResponse,
            temperature=0
        )
        
        semantic_cluster = completion.choices[0].message.parsed.cluster
        if semantic_cluster:
            cluster_id = f"cluster_{type_}_{len(clusters)}"
            for entity in semantic_cluster:
                clusters[entity] = cluster_id
                
    return clusters

def create_knowledge_graph(text: str) -> KnowledgeGraph:
    """Create a knowledge graph from input text"""
    logger.info("Extracting entities...")
    entities = extract_entities(text)
    
    logger.info("Clustering similar entities...")
    clusters = cluster_similar_entities(entities)
    
    logger.info("Extracting relations...")
    relations = extract_relations(text, entities)
    
    # Create nodes
    nodes = []
    for entity, type_ in entities:
        node = KnowledgeGraphNode(
            id=entity,
            label=entity,
            type=type_,
            cluster=clusters.get(entity)
        )
        nodes.append(node)
    
    # Create edges
    edges = []
    for subj, pred, obj in relations:
        edge = KnowledgeGraphEdge(
            source=subj,
            target=obj,
            relation=pred
        )
        edges.append(edge)
    
    return KnowledgeGraph(nodes=nodes, edges=edges)
  
def generate_graph_filename(kg: KnowledgeGraph) -> str:
    """Generate a descriptive filename for the knowledge graph"""
    # Get the first 2-3 most important nodes (prefer organizations/persons)
    important_nodes = sorted(
        [n for n in kg.nodes if n.type in ['organization', 'person', 'concept']],
        key=lambda x: len([e for e in kg.edges if e.source == x.id or e.target == x.id]),
        reverse=True
    )[:3]
    
    # Create a slug from node labels
    name_parts = [node.label.lower().replace(' ', '_') for node in important_nodes]
    base_name = '-'.join(name_parts) if name_parts else 'knowledge_graph'
    
    # Add timestamp to ensure uniqueness
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.json"

def save_knowledge_graph(kg: KnowledgeGraph, filename: str = None):
    """Save knowledge graph to JSON file in the graphs directory"""
    try:
        # Get the absolute path to the _7_knowledge_graph directory
        kg_dir = os.path.dirname(os.path.abspath(__file__))
        graphs_dir = os.path.join(kg_dir, 'graphs')
        
        # Create graphs directory if it doesn't exist
        os.makedirs(graphs_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = generate_graph_filename(kg)
        
        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        full_path = os.path.join(graphs_dir, filename)
        
        # Check write permissions
        if os.path.exists(full_path):
            if not os.access(full_path, os.W_OK):
                raise PermissionError(f"No write permission for {full_path}")
        elif not os.access(graphs_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory {graphs_dir}")
        
        with open(full_path, 'w') as f:
            json.dump(kg.model_dump(), f, indent=2)
        logger.info(f"Saved knowledge graph to {full_path}")
        return filename
    except Exception as e:
        raise RuntimeError(f"Failed to save knowledge graph: {str(e)}")

def main(text: str, output_file: str = None) -> KnowledgeGraph:
    """Main function to process text and generate knowledge graph"""
    logger.info("Generating knowledge graph...")
    kg = create_knowledge_graph(text)
    
    logger.info(f"Created knowledge graph with {len(kg.nodes)} nodes and {len(kg.edges)} edges")
    
    save_knowledge_graph(kg, output_file)
    return kg

if __name__ == "__main__":
    # Example usage
    sample_text = """
    OpenAI was founded by Sam Altman and others in San Francisco. 
    The company developed ChatGPT, which uses large language models for natural language processing.
    Microsoft invested heavily in OpenAI and integrated ChatGPT into their Bing search engine.
    """
    
    kg = main(sample_text)
    
    # Print summary
    print("\nKnowledge Graph Summary:")
    print(f"Nodes: {len(kg.nodes)}")
    for node in kg.nodes:
        print(f"- {node.label} ({node.type})")
    
    print(f"\nEdges: {len(kg.edges)}")
    for edge in kg.edges:
        print(f"- {edge.source} --[{edge.relation}]--> {edge.target}")

export = {
    "default": main,
    "create_knowledge_graph": create_knowledge_graph,
    "save_knowledge_graph": save_knowledge_graph
}
