from pydantic import BaseModel, Field
from anthropic import Anthropic
from typing import List, Dict, Set, Tuple, Optional
import json
import logging
import time
import random
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = Anthropic()
MODEL = "claude-3-5-sonnet-20241022"

current_folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

# Keep the same model classes from the original implementation
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
            except Exception as e:
                if "rate_limit" not in str(e).lower() or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                logger.warning(f"Rate limit hit. Waiting {delay:.2f}s before retry")
                time.sleep(delay)
        
    return wrapper

def validate_tool_output(message, expected_tool_name: str) -> dict:
    """Validate Claude's tool output and return the input data"""
    logger.debug(f"Validating tool output for {expected_tool_name}")
    logger.debug(f"Message content: {message.content}")
    
    if not message.content:
        raise ValueError("Message has no content")
    
    if not message.content[0].type == "tool_use":
        raise ValueError(f"Expected tool_calls but got {message.content[0].type}")
        
    tool_call = message.content[0]
    logger.debug(f"Tool call: {tool_call}")
    
    if tool_call.name != expected_tool_name:
        raise ValueError(f"Expected tool '{expected_tool_name}' but got '{tool_call.name}'")
        
    if not hasattr(tool_call, 'input') or not isinstance(tool_call.input, dict):
        raise ValueError("Tool output missing or invalid input data")
        
    return tool_call.input

@handle_rate_limit
def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract entities and their types from text"""
    try:
        logger.info(f"Attempting to extract entities from text: {text[:100]}...")
        message = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            temperature=0,
            tools=[{
                "name": "extract_entities",
                "description": "Extract entities and their types from text",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["concept", "person", "organization", "location", "event", "object", "other"]
                                    }
                                },
                                "required": ["name", "type"]
                            }
                        }
                    },
                    "required": ["entities"]
                }
            }],
            tool_choice={"type": "tool", "name": "extract_entities"},
            messages=[
                {
                    "role": "user",
                    "content": f"Extract key entities and their types from this text: {text} (max 500 entities)"
                }
            ]
        )
        
        logger.debug(f"Raw Claude response: {message}")
        tool_response = validate_tool_output(message, "extract_entities")
        logger.debug(f"Validated tool response: {tool_response}")
        
        if "entities" not in tool_response:
            raise ValueError("Tool response missing 'entities' field")
            
        entities = []
        for entity in tool_response["entities"]:
            if not isinstance(entity, dict) or "name" not in entity or "type" not in entity:
                logger.warning(f"Skipping invalid entity: {entity}")
                continue
            entities.append((entity["name"], entity["type"]))
        
        logger.info(f"Successfully extracted {len(entities)} entities: {entities}")    
        return entities
        
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}", exc_info=True)  # Added exc_info for stack trace
        return []

@handle_rate_limit
def extract_relations(text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """Extract relationships between entities"""
    if not entities:
        logger.warning("No entities provided for relation extraction")
        return []
        
    try:
        entity_list = [e[0] for e in entities]
        entity_str = ", ".join(entity_list)
        logger.info(f"Attempting to extract relations between entities: {entity_str}")
        
        message = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            temperature=0,
            tools=[{
                "name": "extract_relations",
                "description": "Extract relations between entities",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "description": "An array of relations extracted from the text, where each relation is represented as a subject-predicate-object triple.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "subject": {
                                        "type": "string",
                                        "description": "The entity that is the subject of the relationship. Must be one of the provided entities."
                                    },
                                    "predicate": {
                                        "type": "string",
                                        "description": "A concise (1-3 words) description of the relation between the subject and object."
                                    },
                                    "object": {
                                        "type": "string",
                                        "description": "The entity that is the object of the relation. Must be one of the provided entities."
                                    }
                                },
                                "required": ["subject", "predicate", "object"]
                            }
                        }
                    },
                    "required": ["relations"]
                }
            }],
            tool_choice={"type": "tool", "name": "extract_relations"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Extract knowledge graph relations from this text (max 1000 relations): {text}\n\n---\n\nUse these entities: {entity_str}
                            """
                        }
                    ]
                }
            ]
        )
        
        logger.debug(f"Raw Claude response: {message}")
        tool_response = validate_tool_output(message, "extract_relations")
        logger.debug(f"Validated tool response: {tool_response}")
        
        # Check if we have the expected structure
        if not isinstance(tool_response, dict):
            logger.error(f"Tool response is not a dictionary: {tool_response}")
            return []
            
        if "relations" not in tool_response:
            logger.error(f"Tool response missing 'relations' field. Response: {tool_response}")
            return []
            
        relations = []
        entity_set = set(entity_list)  # For faster lookups
        
        for rel in tool_response["relations"]:
            if not isinstance(rel, dict):
                logger.warning(f"Invalid relation format: {rel}")
                continue
                
            subj = rel.get("subject", "").strip()
            pred = rel.get("predicate", "").strip()
            obj = rel.get("object", "").strip()
            
            logger.debug(f"Processing relation: {subj} --[{pred}]--> {obj}")
            
            # Validate that subject and object are in our entity list
            if not subj or not obj or not pred:
                logger.warning(f"Missing component in relation: ({subj}, {pred}, {obj})")
                continue
            if subj not in entity_set:
                logger.warning(f"Subject '{subj}' not in entity list")
                continue
            if obj not in entity_set:
                logger.warning(f"Object '{obj}' not in entity list")
                continue
                
            relations.append((subj, pred, obj))
            
        logger.info(f"Successfully extracted {len(relations)} relations: {relations}")
        return relations
        
    except Exception as e:
        logger.error(f"Error extracting relations: {str(e)}", exc_info=True)
        return []

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
            
        message = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            temperature=0,
            tools=[{
                "name": "cluster_entities",
                "description": "Group semantically similar entities",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "clusters": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "required": ["clusters"]
                }
            }],
            tool_choice={"type": "tool", "name": "cluster_entities"},
            messages=[
                {
                    "role": "user",
                    "content": f"""Group these entities into clusters based on semantic similarity: {', '.join(type_entities)}
                    Consider variations in tenses, plural forms, stem forms, capitalization, and semantic meaning."""
                }
            ]
        )
        
        tool_response = message.content[0].input
        
        for cluster_idx, cluster in enumerate(tool_response["clusters"]):
            cluster_id = f"cluster_{type_}_{len(clusters) + cluster_idx}"
            for entity in cluster:
                clusters[entity] = cluster_id
                
    return clusters

@handle_rate_limit
def cluster_similar_relations(relations: List[Tuple[str, str, str]]) -> Dict[str, str]:
    """Group similar relation predicates and standardize them"""
    if not relations:
        return {}
        
    try:
        predicates = list(set(rel[1] for rel in relations))
        
        if len(predicates) < 2:
            return {predicates[0]: predicates[0]} if predicates else {}
        
        message = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            temperature=0,
            tools=[{
                "name": "cluster_relations",
                "description": "Group and standardize similar relation predicates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "standardized_predicates": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "string",
                                "description": "The standardized form for this predicate"
                            },
                            "description": "Map of original predicates to their standardized forms"
                        }
                    },
                    "required": ["standardized_predicates"]
                }
            }],
            tool_choice={"type": "tool", "name": "cluster_relations"},
            messages=[
                {
                    "role": "user",
                    "content": f"""Group these relation predicates and provide a standardized form for each: {', '.join(predicates)}
                    Consider:
                    - Synonyms (e.g., 'created'/'developed'/'built')
                    - Tense variations (e.g., 'invests'/'invested')
                    - Style variations (e.g., 'is part of'/'belongs to')
                    Return a mapping of original predicates to their standardized forms."""
                }
            ]
        )
        
        tool_response = validate_tool_output(message, "cluster_relations")
        
        if "standardized_predicates" not in tool_response:
            raise ValueError("Tool response missing 'standardized_predicates' field")
            
        standardized = tool_response["standardized_predicates"]
        if not isinstance(standardized, dict):
            raise ValueError("Invalid standardized_predicates format")
            
        # Validate the standardized predicates
        validated = {}
        for orig, std in standardized.items():
            if not isinstance(orig, str) or not isinstance(std, str):
                continue
            validated[orig.strip()] = std.strip()
            
        return validated
        
    except Exception as e:
        logger.error(f"Error clustering relations: {str(e)}")
        return {pred: pred for pred in predicates}  # Fall back to no clustering

def create_knowledge_graph(text: str) -> KnowledgeGraph:
    """Create a knowledge graph from input text"""
    try:
        logger.info(f"Starting knowledge graph creation for text: {text[:100]}...")
        
        logger.info("Extracting entities...")
        entities = extract_entities(text)
        if not entities:
            logger.warning("No entities found in text")
            return KnowledgeGraph()
        logger.info(f"Found {len(entities)} entities")
        
        logger.info("Clustering similar entities...")
        clusters = cluster_similar_entities(entities)
        logger.info(f"Created {len(set(clusters.values()))} clusters")
        
        logger.info("Extracting relations...")
        relations = extract_relations(text, entities)
        if not relations:
            logger.warning("No relations found between entities")
        logger.info(f"Found {len(relations)} relations")
        
        logger.info("Standardizing relations...")
        standardized_predicates = cluster_similar_relations(relations)
        logger.info(f"Standardized {len(standardized_predicates)} predicates")
        
        # Create nodes and edges with additional validation
        nodes = []
        node_ids = set()
        for entity, type_ in entities:
            if not entity or not type_:
                logger.warning(f"Skipping invalid entity: ({entity}, {type_})")
                continue
            if entity in node_ids:
                logger.debug(f"Skipping duplicate entity: {entity}")
                continue
            node_ids.add(entity)
            
            nodes.append(KnowledgeGraphNode(
                id=entity,
                label=entity,
                type=type_,
                cluster=clusters.get(entity)
            ))
        
        edges = []
        for subj, pred, obj in relations:
            if not all([subj, pred, obj]):
                logger.warning(f"Skipping invalid relation: ({subj}, {pred}, {obj})")
                continue
            if subj not in node_ids or obj not in node_ids:
                logger.warning(f"Skipping relation with missing nodes: {subj} -> {obj}")
                continue
                
            edges.append(KnowledgeGraphEdge(
                source=subj,
                target=obj,
                relation=standardized_predicates.get(pred, pred)
            ))
        
        logger.info(f"Created knowledge graph with {len(nodes)} nodes and {len(edges)} edges")
        return KnowledgeGraph(nodes=nodes, edges=edges)
        
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {str(e)}", exc_info=True)
        return KnowledgeGraph()  # Return empty graph on error

# Keep the same utility functions from the original implementation
def generate_graph_filename(kg: KnowledgeGraph) -> str:
    """Generate a descriptive filename for the knowledge graph"""
    important_nodes = sorted(
        [n for n in kg.nodes if n.type in ['organization', 'person', 'concept']],
        key=lambda x: len([e for e in kg.edges if e.source == x.id or e.target == x.id]),
        reverse=True
    )[:3]
    
    name_parts = [node.label.lower().replace(' ', '_') for node in important_nodes]
    base_name = '-'.join(name_parts) if name_parts else 'knowledge_graph'
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.json"

def save_knowledge_graph(kg: KnowledgeGraph, filename: str = None):
    """Save knowledge graph to JSON file in the graphs directory"""
    try:
        kg_dir = os.path.dirname(os.path.abspath(__file__))
        graphs_dir = os.path.join(kg_dir, 'graphs')
        
        os.makedirs(graphs_dir, exist_ok=True)
        
        if filename is None:
            filename = generate_graph_filename(kg)
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        full_path = os.path.join(graphs_dir, filename)
        
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
    Overview and Uses of Tylenol\nTylenol, generically known as acetaminophen, is a widely used medication primarily indicated for pain relief and fever reduction. It is commonly employed to manage mild to moderate pain conditions, including headaches, muscle aches, arthritis, toothaches, backaches, and menstrual cramps. Tylenol is also frequently used to reduce fever caused by infections such as the flu or the common cold. Unlike nonsteroidal anti-inflammatory drugs (NSAIDs) such as ibuprofen or aspirin, acetaminophen does not possess significant anti-inflammatory properties, making it preferable in cases where inflammation is not a primary concern. Other drugs used to treat similar symptoms include NSAIDs like ibuprofen and naproxen, as well as combination medications that pair acetaminophen with other compounds, such as codeine or caffeine, for enhanced pain relief.\nChemical Composition and Related Drugs\nAcetaminophen is chemically classified as a para-aminophenol derivative with the formula C8H9NO2C_8H_9NO_2C8\u200bH9\u200bNO2\u200b. It works primarily by inhibiting cyclooxygenase (COX) enzymes in the central nervous system, reducing the production of prostaglandins that mediate pain and fever. Structurally similar drugs include phenacetin, which was historically used for similar purposes but is no longer widely available due to safety concerns, and other analgesic-antipyretic agents. Unlike NSAIDs, acetaminophen lacks a carboxylic acid group, which contributes to its minimal gastrointestinal irritation compared to drugs like aspirin. However, acetaminophen\u2019s similarity to phenacetin highlights a shared risk for liver toxicity at high doses, necessitating careful adherence to dosing guidelines.\nSide Effects, Interactions, and Warnings\nTylenol is generally well-tolerated when used as directed but can cause side effects such as nausea, rash, or, rarely, severe allergic reactions. The most serious risk associated with acetaminophen is liver damage, particularly at doses exceeding 4,000 mg per day or when combined with alcohol or other hepatotoxic drugs. It interacts poorly with medications like warfarin, potentially increasing the risk of bleeding, and should not be combined with other acetaminophen-containing products to avoid overdose. However, Tylenol is considered a safer alternative to NSAIDs for individuals at risk of gastrointestinal bleeding or kidney issues. Important warnings include its contraindication in patients with severe liver disease and the need for caution in populations like pregnant women or individuals with chronic alcohol use. Other names for acetaminophen include paracetamol, Panadol, and APAP (short for N-acetyl-p-aminophenol), a designation frequently used in prescription labeling.\nOverview and Uses of Advil\nAdvil, the brand name for ibuprofen, is a widely used nonsteroidal anti-inflammatory drug (NSAID) indicated for relieving pain, reducing inflammation, and lowering fever. It is commonly employed to manage a variety of conditions, including headaches, dental pain, menstrual cramps, muscle aches, arthritis, and mild to moderate injuries such as sprains or strains. It is also effective in alleviating symptoms of inflammatory conditions like rheumatoid arthritis and osteoarthritis. As a fever reducer, Advil is often used in cases of flu or common cold. Other medications that treat similar symptoms include acetaminophen (Tylenol), which is not anti-inflammatory, and NSAIDs such as naproxen (Aleve) and aspirin, which share ibuprofen\u2019s anti-inflammatory and analgesic properties.\nChemical Composition and Related Drugs\nIbuprofen is a propionic acid derivative with the chemical formula C13H18O2C_{13}H_{18}O_2C13\u200bH18\u200bO2\u200b. It functions by inhibiting cyclooxygenase (COX) enzymes, primarily COX-1 and COX-2, thereby reducing the synthesis of prostaglandins that mediate pain, inflammation, and fever. Drugs with similar chemical compositions and mechanisms of action include other NSAIDs, such as naproxen, diclofenac, and ketoprofen, which are also derivatives of organic acids. Ibuprofen is distinguished by its relatively short half-life compared to naproxen, making it suitable for acute symptom management but requiring more frequent dosing for chronic conditions. Its racemic mixture contains both active and inactive enantiomers, with the S-enantiomer responsible for its pharmacological effects.\nSide Effects, Interactions, and Warnings\nWhile Advil is generally safe when used as directed, it can cause side effects, particularly with prolonged use or at higher doses. Common side effects include stomach upset, nausea, and heartburn. More serious risks include gastrointestinal bleeding, ulcers, kidney damage, and cardiovascular events like heart attack or stroke, especially in patients with pre-existing conditions. Ibuprofen should be used cautiously in combination with anticoagulants such as warfarin or aspirin due to an increased risk of bleeding, and it can interact poorly with medications like diuretics and ACE inhibitors, potentially diminishing their effectiveness. Conversely, it is often used alongside acetaminophen for enhanced pain relief, as the two drugs work through different mechanisms. Important warnings include avoiding ibuprofen in individuals with a history of peptic ulcers, severe kidney disease, or hypersensitivity to NSAIDs. Pregnant women, particularly in the third trimester, should avoid ibuprofen due to the risk of fetal harm. Other names for ibuprofen include Motrin, Brufen, and generic ibuprofen formulations available globally.
    """
    # sample_text = """
    # OpenAI was founded by Sam Altman and others in San Francisco. 
    # The company developed ChatGPT, which uses large language models for natural language processing.
    # Microsoft invested heavily in OpenAI and integrated ChatGPT into their Bing search engine.
    # """
    
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
