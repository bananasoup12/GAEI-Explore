import random
import json
import os
from dataclasses import dataclass
from neo4j import GraphDatabase
import re
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import time
import logging
from tqdm import tqdm
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GraphConfig:
    """Configuration for graph traversal parameters."""
    uri: str
    username: str
    password: str
    top_n: int = 3
    random_nodes: int = 5
    mode: str = 'random'
    specific_node: str = ""
    depth: int = 30
    bounds: Tuple[int, int] = (0, 100)
    min_nodes: int = 4

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.mode not in ['random', 'specific']:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.mode == 'specific' and not self.specific_node:
            raise ValueError("Specific mode requires a specific_node")
        if not (0 <= self.bounds[0] <= self.bounds[1] <= 100):
            raise ValueError(f"Invalid bounds: {self.bounds}")
        if self.depth < 1:
            raise ValueError(f"Invalid depth: {self.depth}")

class RelationshipRankings:
    """Manages relationship rankings with validation."""
    _rankings = {
        "/r/IsA": 1,
        "/r/InstanceOf": 2,
        "/r/PartOf": 3,
        "/r/HasA": 4,
        "/r/HasProperty": 5,
        "/r/SimilarTo": 6,
        "/r/RelatedTo": 7,
        "/r/CapableOf": 8,
        "/r/UsedFor": 9,
        "/r/Causes": 10,
        "/r/ReceivesAction": 11,
        "/r/HasPrerequisite": 12,
        "/r/MotivatedByGoal": 13,
        "/r/HasSubevent": 14,
        "/r/HasFirstSubevent": 15,
        "/r/HasLastSubevent": 16,
        "/r/CausesDesire": 17,
        "/r/Desires": 18,
        "/r/Entails": 19,
        "/r/MannerOf": 20,
        "/r/SymbolOf": 21,
        "/r/EtymologicallyRelatedTo": 22,
        "/r/EtymologicallyDerivedFrom": 23,
        "/r/DerivedFrom": 24,
        "/r/FormOf": 25,
        "/r/DefinedAs": 26,
        "/r/CreatedBy": 27,
        "/r/MadeOf": 28,
        "/r/AtLocation": 29,
        "/r/LocatedNear": 30,
        "/r/DistinctFrom": 31,
        "/r/NotCapableOf": 32,
        "/r/NotDesires": 33,
        "/r/NotHasProperty": 34,
        "/r/Synonym": 35,
        "/r/Antonym": 36,
        "/r/HasContext": 37,
        "/r/dbpedia/occupation": 38,
        "/r/dbpedia/genre": 39,
        "/r/dbpedia/field": 40,
        "/r/dbpedia/product": 41,
        "/r/dbpedia/knownFor": 42,
        "/r/dbpedia/genus": 43,
        "/r/dbpedia/language": 44,
        "/r/dbpedia/capital": 45,
        "/r/dbpedia/leader": 46,
        "/r/dbpedia/influencedBy": 47
    }

    @classmethod
    def get_rank(cls, relationship_type: str) -> int:
        """Get ranking with default handling."""
        return cls._rankings.get(relationship_type, float('inf'))

class PathGenerator(ABC):
    """Abstract base class for path generation strategies."""
    @abstractmethod
    def generate_paths(self, graph: 'Neo4jGraph', num_paths: int) -> List[Dict[str, Any]]:
        pass

class RandomPathGenerator(PathGenerator):
    """Implements random path generation strategy."""
    def generate_paths(self, graph: 'Neo4jGraph', num_paths: int) -> List[Dict[str, Any]]:
        paths = []
        attempts = 0
        max_attempts = num_paths * 2
        
        start_time = time.time()
        progress_bar = tqdm(total=num_paths, desc="Generating Paths", unit="path")

        with graph.driver.session() as session:
            
            while len(paths) < num_paths and attempts < max_attempts:
                attempts += 1
                start_node = graph.get_start_node(session)

                if not start_node:
                    continue

                try:
                    path_data = graph.explore_path(session, start_node, len(paths) + 1)

                    if (path_data and 
                        graph.config.min_nodes <= len(path_data['nodes']) <= graph.config.depth and
                        str(path_data['nodes']) not in graph.visited_paths):

                        graph.visited_paths.add(str(path_data['nodes']))
                        path_data['nodes'] = graph.filter_nodes(path_data['nodes'], session)
                        paths.append(path_data)
                        progress_bar.update(1)

                        # Estimate remaining time
                        elapsed_time = time.time() - start_time
                        avg_time_per_path = elapsed_time / len(paths) if paths else 0
                        remaining_time = avg_time_per_path * (num_paths - len(paths))
                        progress_bar.set_postfix(eta=f"{remaining_time:.2f}s")

                except Exception as e:
                    logger.error(f"Error generating path: {e}")
                    continue

        progress_bar.close()
        return paths

class Neo4jGraph:
    """Enhanced Neo4j graph interface using URIs for node identification."""
    def __init__(self, config: GraphConfig):
        self.config = config
        self.config.validate()
        self.driver = GraphDatabase.driver(
            config.uri, 
            auth=(config.username, config.password)
        )
        self.exploration_rate = 0.1
        self.visited_paths: Set[str] = set()
        self._verify_connection()

    def find_matching_node(self, session, node_pattern: str) -> Optional[str]:
        """
        Find the best matching node URI for a given pattern.
        
        Args:
            session: Neo4j session
            node_pattern: Input pattern to match
            
        Returns:
            Best matching node URI or None if no match found
        """
        try:
            pattern = node_pattern.lower().replace(' ', '_')
            
            query = """
            MATCH (n)
            WHERE 
            toLower(n.uri) CONTAINS $pattern
            OR toLower(n.uri) = $exact_match
            RETURN n.uri AS uri
            ORDER BY 
            CASE 
                WHEN toLower(n.uri) = $exact_match THEN 0
                WHEN toLower(n.uri) CONTAINS $pattern THEN 1
                ELSE 2
            END,
            size(n.uri)
            LIMIT 1
            """
            
            result = session.run(
                query,
                pattern=pattern,
                exact_match=pattern
            )
            record = result.single()
            
            if record:
                # logger.info(f"Found matching node: {record['uri']} for input: {node_pattern}")
                return record['uri']
            else:
                # logger.warning(f"No matching node found for: {node_pattern}")
                return None
                
        except Exception as e:
            logger.error(f"Error in find_matching_node: {e}")
            return None

    def get_start_node(self, session) -> Optional[str]:
        """Get the starting node URI based on the current mode."""
        try:
            if self.config.mode == 'specific':
                matched_node = self.find_matching_node(session, self.config.specific_node)
                if matched_node:
                    return matched_node
                logger.warning(f"Specified node '{self.config.specific_node}' not found. Falling back to random selection.")
                self.config.mode = 'random'
            
            if self.config.mode == 'random':
                result = session.run("""
                    MATCH (n)
                    RETURN n.uri AS uri
                    ORDER BY RAND()
                    LIMIT 1
                """)
                record = result.single()
                return record['uri'] if record else None
            
            return None
        except Exception as e:
            logger.error(f"Error getting start node: {e}")
            return None

    def get_next_nodes(self, session, current_node: str, visited_nodes: Set[str]) -> List[Dict[str, Any]]:
        """Get all valid next nodes from the current node."""
        try:
            query = """
            MATCH (a {uri: $node})-[r]->(b)
            WHERE NOT b.uri IN $visited
            RETURN b.uri AS end_node, 
                   r.type AS relationship_type
            """
            result = session.run(query, node=current_node, visited=list(visited_nodes))
            return [{
                'end_node': record['end_node'],
                'relationship_type': record['relationship_type']
            } for record in result]
        except Exception as e:
            logger.error(f"Error getting next nodes: {e}")
            return []

    def choose_next_node(self, nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Choose the next node based on probabilistic weighting of relationship rankings."""
        if not nodes:
            return None
        
        try:
            # Compute relationship ranks and apply softmax for weighting
            ranks = [RelationshipRankings.get_rank(node['relationship_type']) for node in nodes]
            max_rank = max(ranks)  # Normalize ranks
            weights = [math.exp(-rank) for rank in ranks]  # Softmax-like transformation
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            
            # Introduce exploration: with probability `self.exploration_rate`, choose randomly
            if random.random() < self.exploration_rate:
                return random.choice(nodes)
            
            # Otherwise, choose a node based on computed probabilities
            return random.choices(nodes, weights=probabilities, k=1)[0]
        except Exception as e:
            logger.error(f"Error choosing next node: {e}")
            return None


    def explore_path(self, session, start_node: str, path_id: int) -> Dict[str, Any]:
        """Explore a path through the graph starting from the given node."""
        visited_nodes: Set[str] = set()
        path_nodes: List[Dict[str, Any]] = []
        current_node = start_node
        depth = 0

        try:
            while len(path_nodes) < self.config.depth:
                if current_node in visited_nodes and random.random() > 0.1:  # 90% chance to continue forward
                    break

                node_data = {
                    'id': f"{path_id}_{current_node}",
                    'uri': current_node
                }

                if path_nodes:  # Not the first node
                    prev_node = path_nodes[-1]['uri']
                    relationship = self.get_relationship_data(session, prev_node, current_node)
                    node_data['relationship'] = relationship

                path_nodes.append(node_data)
                visited_nodes.add(current_node)

                # 10% chance to backtrack
                if len(visited_nodes) > 1 and random.random() < (depth/100.0):
                    current_node = random.choice(list(visited_nodes - {current_node}))
                    depth = 0
                    continue  # Restart the loop with the backtracked node

                next_nodes = self.get_next_nodes(session, current_node, visited_nodes)
                next_node = self.choose_next_node(next_nodes)

                if not next_node:
                    next_node = random.choice(list(visited_nodes - {current_node}))
                    break
                    
                current_node = next_node['end_node']
                depth = depth + 1
            return {
                'path_id': path_id,
                'nodes': path_nodes
            }
        except Exception as e:
            logger.error(f"Error exploring path: {e}")
            return {'path_id': path_id, 'nodes': path_nodes}

    def get_relationship_data(self, session, start_node: str, end_node: str) -> Dict[str, Any]:
        """Get relationship type between two nodes."""
        try:
            query = """
            MATCH (a {uri: $start_node})-[r]->(b {uri: $end_node})
            RETURN r.type AS type
            """
            result = session.run(query, start_node=start_node, end_node=end_node)
            record = result.single()
            if record:
                return {
                    'start': start_node,
                    'end': end_node,
                    'type': record['type'],
                    'rank': RelationshipRankings.get_rank(record['type'])
                }
        except Exception as e:
            logger.error(f"Error getting relationship data: {e}")
        
        return {
            'start': start_node,
            'end': end_node,
            'type': None,
            'rank': float('inf')
        }

    def _verify_connection(self) -> None:
        """Verify Neo4j connection is working."""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) RETURN n LIMIT 1")
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    def filter_nodes(self, nodes: List[Dict[str, Any]], session, 
                    root_node: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Filter nodes based only on relationship rankings."""
        try:
            if root_node:
                nodes.insert(0, root_node)

            # Sort by relationship rank if available
            nodes.sort(key=lambda x: x.get('relationship', {}).get('rank', float('inf')))
            
            # Get top nodes
            top_nodes = nodes[:self.config.top_n]
            
            # Add random nodes
            remaining_nodes = [node for node in nodes if node not in top_nodes]
            if remaining_nodes and self.config.random_nodes > 0:
                random_nodes = random.sample(remaining_nodes, 
                                           min(self.config.random_nodes, len(remaining_nodes)))
                return top_nodes + random_nodes
            
            return top_nodes
        except Exception as e:
            logger.error(f"Error filtering nodes: {e}")
            return nodes

def create_prompt(
    path: Dict[str, Any], 
    user_profile: str, 
    question: str = "", 
    output_level: str = "expert"
) -> str:
    """
    Create a prompt that critically evaluates the feasibility and novelty of ideas while ensuring alignment with the user's capabilities.

    Args:
        path: Dictionary containing nodes and relationships. Can be empty.
        user_profile: Description of the user's capabilities (e.g., "a high school student with basic programming skills",
                      "a large tech corporation with billions in R&D funding").
        question: Optional question to guide the idea generation. Default empty string.
        output_level: Level of explanation detail (e.g., "expert", "student", "5-year-old"). Default is "expert".

    Returns:
        str: Generated prompt incorporating feasibility based on the user's profile and explanation level.
    """

    json_structure = '''
    {
        "idea": "Multi-line formatted description of the idea",
        "explanation": "Explanation of the idea tailored to the chosen output level",
        "analysis": {
            "novelty": {
                "score": "1-100",
                "justification": "Evidence-based assessment of the idea's originality, citing existing approaches and identifying genuinely new elements"
            },
            "usefulness": {
                "score": "1-100",
                "justification": "Analysis of specific problems addressed and advantages over current solutions"
            },
            "feasibility": {
                "score": "1-100",
                "justification": "Evaluation of technical barriers, resource requirements, and implementation challenges",
                "prerequisites": ["List of key technologies or capabilities required"],
                "alignment_with_user": "Explanation of how the idea fits the user's expertise, resources, and constraints"
            },
            "impact": {
                "score": "1-100",
                "justification": "Assessment of potential benefits weighed against limitations",
                "potential_applications": ["List of specific use cases"]
            }
        },
        "composite_score": "Weighted average with feasibility given higher importance. In the case the usual profle is closer to an innovator or researcher, the feasibility should take lower importance.",
        "critical_evaluation": "Summary of strengths and weaknesses, identifying potential misconceptions or overreaching claims",
        "related_fields": ["List of 3 most relevant disciplines or subjects"]
    }
    '''

    base_prompt = f"""
    {"Consider the following question: " + question if question else "Generate and critically evaluate an idea or insight"}

    {f"Given the following nodes and relationships, " if path else ""}generate an idea that can be rigorously evaluated for its scientific and practical merit.

    The idea should be **feasible** given the user's background:
    - **User profile**: {user_profile}
    - **Available expertise, resources, and constraints should be considered.**

    Tailor the explanation of the idea based on the specified output level:
    - **Expert**: Provide a precise, technical breakdown with domain-specific terminology.
    - **Student**: Use clear, structured explanations that assume intermediate knowledge.
    - **5-year-old**: Use very simple language and analogies that make the idea easy to grasp.

    Your analysis should:
    1. Distinguish between genuinely novel elements and those already established in the field.
    2. Identify specific prerequisites and implementation challenges.
    3. Assess feasibility **relative to the user's profile** rather than in absolute terms.
    4. Suggest possible adaptations or scaled-down versions if necessary.
    5. Avoid:
       - Misapplying scientific or mathematical concepts.
       - Making scale mismatches (e.g., suggesting billion-dollar projects for individuals).
       - Creating false equivalences between unrelated phenomena.

    For feasibility, explicitly address:
    - The current state of technology.
    - Required expertise and resources.
    - Any fundamental limitations.
    - How well the idea aligns with the user's profile.

    Provide a structured JSON output in the following format:
    {json_structure}

    Your evaluation should be balanced, highlighting both strengths and limitations while remaining grounded in realistic execution potential.

    The explanation of the idea should be tailored to the requested **output level**: "{output_level}".

    ### **Examples of Evaluated Ideas:**
    
    #### **Example 1 - Fundamentally Infeasible Idea (Score: 30)**
    {{"idea": "Quantum teleportation for human travel",
      "explanation": "For a 5-year-old: Imagine if you could snap your fingers and appear anywhere in the world instantly. Scientists can do this with tiny particles, but not with people.",
      "analysis": {{
        "novelty": {{"score": 65, "justification": "Quantum teleportation exists for quantum states, but not for macroscopic objects."}},
        "usefulness": {{"score": 90, "justification": "Would revolutionize travel if possible."}},
        "feasibility": {{"score": 5, "justification": "Fundamentally impossible due to quantum mechanics constraints.",
                        "prerequisites": ["Rewriting laws of physics", "Quantum entanglement on an unprecedented scale"],
                        "alignment_with_user": "Not feasible for any entity at current technological level."}},
        "impact": {{"score": 95, "justification": "Would reshape human civilization but remains science fiction.",
                    "potential_applications": ["Instant global transport", "Deep space travel"]}}
      }},
      "composite_score": "30 (heavily penalized by feasibility constraints)",
      "critical_evaluation": "Misunderstands quantum teleportation—it transfers information, not matter.",
      "related_fields": ["Quantum physics", "Transportation", "Information theory"]}}
    
    #### **Example 2 - Cost-Prohibitive but Technically Feasible Idea (Score: 65)**
    {{"idea": "Global direct air carbon capture network",
      "explanation": "For a student: This is like giant vacuum cleaners for the sky, sucking out bad air (CO2) to fight climate change. The problem is that it costs way too much right now.",
      "analysis": {{
        "novelty": {{"score": 45, "justification": "Direct air capture (DAC) exists, but global deployment is novel."}},
        "usefulness": {{"score": 85, "justification": "Addresses climate change directly."}},
        "feasibility": {{"score": 60, "justification": "Technically possible but prohibitively expensive at current costs.",
                        "prerequisites": ["Cheaper DAC technology", "Vast land areas for solar power"],
                        "alignment_with_user": "Feasible for large governments or multinational corporations, but not individuals."}},
        "impact": {{"score": 80, "justification": "Significant climate benefits but requires global coordination.",
                    "potential_applications": ["Climate mitigation", "Carbon credit markets"]}}
      }},
      "composite_score": "65 (limited by cost barriers despite technical feasibility)",
      "critical_evaluation": "Economic feasibility is the main bottleneck, requiring policy support.",
      "related_fields": ["Climate engineering", "Renewable energy", "Chemical engineering"]}}

    #### **Example 3 - Highly Practical Idea (Score: 82)**
    {{"idea": "AI-powered predictive maintenance for infrastructure",
      "explanation": "For an expert: This involves integrating sensor networks with machine learning models to detect early signs of failure in bridges, power grids, and factories.",
      "analysis": {{
        "novelty": {{"score": 65, "justification": "Combines existing AI and physical models in a new way."}},
        "usefulness": {{"score": 90, "justification": "Reduces infrastructure failures and maintenance costs."}},
        "feasibility": {{"score": 85, "justification": "All required technologies exist and can be integrated.",
                        "prerequisites": ["Sensor networks", "Machine learning", "Industry partnerships"],
                        "alignment_with_user": "Feasible for tech companies or research institutions with data access."}},
        "impact": {{"score": 85, "justification": "Could significantly improve infrastructure reliability.",
                    "potential_applications": ["Power grids", "Transportation", "Manufacturing"]}}
      }},
      "composite_score": "82 (strong across all dimensions with high feasibility)",
      "critical_evaluation": "The main challenge is domain adaptation, but modular deployment is possible.",
      "related_fields": ["Machine learning", "IoT sensor networks", "Infrastructure engineering"]}}
    """

    return base_prompt



def main():
    """Enhanced main function with better error handling and configuration."""
    try:
        # Load configuration
        config = GraphConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "novanova"),
            mode='specific',
            specific_node="Mobile Home",
            depth=30,
            bounds=(0, 100),
            min_nodes=5
        )

        # Create output directory
        Path('paths').mkdir(exist_ok=True)

        # Initialize graph and path generator
        graph = Neo4jGraph(config)
        path_generator = RandomPathGenerator()

        # Generate paths
        paths = path_generator.generate_paths(graph, num_paths=200)
        
        # Save results
        for path in paths:
            file_path = Path('paths') / f"path_{path['path_id']}.json"
            with open(file_path, 'w') as f:
                json.dump({
                    'prompt': create_prompt(path, "Construction Entrepreneur", "I want ideas around a business based on auxilary dwelling units in Massachusetts?", "Construction Entrepreneur"),
                    'path': path
                }, f, indent=4)

        logger.info(f"Successfully generated {len(paths)} paths")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
    finally:
        if 'graph' in locals():
            graph.close()

if __name__ == "__main__":
    main()








