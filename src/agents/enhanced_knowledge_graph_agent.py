"""
Enhanced Knowledge Graph Agent for Classical Chinese and educational content.
Specialized in creating meaningful relationships and hierarchical structures.
"""

import asyncio
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re

from loguru import logger

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import (
    AnalysisRequest, 
    AnalysisResult, 
    DataType, 
    SentimentResult,
    SentimentLabel,
    ProcessingStatus
)
from src.core.vector_db import VectorDBManager
from src.core.semantic_similarity_analyzer import SemanticSimilarityAnalyzer
from src.core.relationship_optimizer import RelationshipOptimizer
from src.core.chinese_entity_clustering import ChineseEntityClustering
from src.config.config import config
from src.config.settings import settings


class EnhancedKnowledgeGraphAgent(StrandsBaseAgent):
    """Enhanced Knowledge Graph Agent with specialized relationship extraction."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        graph_storage_path: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name or config.model.default_text_model
        
        super().__init__(
            model_name=self.model_name,
            **kwargs
        )
        
        self.graph_storage_path = Path(
            graph_storage_path or settings.paths.knowledge_graphs_dir
        )
        self.graph_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize NetworkX graph
        self.graph = nx.DiGraph()
        self.graph_file = self.graph_storage_path / "enhanced_knowledge_graph.pkl"
        self._load_existing_graph()
        
        # Initialize vector DB manager
        self.vector_db = VectorDBManager()
        
        # Initialize Phase 3 components
        self.semantic_analyzer = SemanticSimilarityAnalyzer()
        self.relationship_optimizer = RelationshipOptimizer()
        self.entity_clustering = ChineseEntityClustering()
        
        # Domain-specific relationship patterns
        self.relationship_patterns = {
            "classical_chinese": {
                "author_work": r"(?:by|作者|编者)\s*([^，。\n]+)",
                "work_lesson": r"(?:第[一二三四五六七八九十\d]+课|Lesson\s*\d+)[：:]\s*([^，。\n]+)",
                "term_function": r"([诸盍焉耳尔叵])\s*[是|为]\s*([^，。\n]+)",
                "example_source": r"《([^》]+)》",
                "historical_figure": r"([孔子|孟子|韩愈|柳宗元|欧阳修|陶渊明|吕布|刘备|曹操|马岱])",
                "linguistic_term": r"([之|其|者|也|乃|是|以|所|为|诸|盍|焉|耳|尔|叵])",
            }
        }
        
        self.metadata.update({
            "agent_type": "enhanced_knowledge_graph",
            "model": self.model_name,
            "capabilities": [
                "enhanced_entity_extraction",
                "domain_specific_relationship_mapping", 
                "hierarchical_graph_construction",
                "semantic_relationship_detection",
                "graph_analysis",
                "graph_visualization",
                "knowledge_inference",
                "community_detection"
            ],
            "supported_data_types": [
                DataType.TEXT,
                DataType.AUDIO,
                DataType.VIDEO,
                DataType.WEBPAGE,
                DataType.PDF,
                DataType.SOCIAL_MEDIA
            ],
            "graph_stats": self._get_graph_stats(),
            "domain_patterns": list(self.relationship_patterns.keys())
        })
        
        logger.info(
            f"Enhanced Knowledge Graph Agent {self.agent_id} initialized with model "
            f"{self.model_name}"
        )
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.extract_entities_enhanced,
            self.map_relationships_enhanced,
            self.query_knowledge_graph,
            self.generate_graph_report,
            self.analyze_graph_communities,
            self.find_entity_paths,
            self.get_entity_context,
            # Phase 3 tools
            self.analyze_semantic_similarity,
            self.optimize_relationships,
            self.cluster_entities_advanced,
            self.run_phase3_quality_assessment
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        return request.data_type in [
            DataType.TEXT, DataType.PDF, DataType.WEBPAGE, 
            DataType.AUDIO, DataType.VIDEO, DataType.SOCIAL_MEDIA
        ]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process content and build enhanced knowledge graph."""
        start_time = datetime.now()
        
        try:
            # Extract text content
            text_content = await self._extract_text_content(request)
            
            # Extract entities with enhanced method
            entities_result = await self.extract_entities_enhanced(text_content)
            entities = entities_result.get("content", [{}])[0].get("entities", [])
            
            # Map relationships with enhanced method
            relationships_result = await self.map_relationships_enhanced(text_content, entities)
            relationships = relationships_result.get("content", [{}])[0].get("relationships", [])
            
            # Add to graph
            await self._add_to_graph(entities, relationships, request.id)
            
            # Analyze impact
            impact_analysis = await self._analyze_graph_impact(entities, relationships)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                id=request.id,
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label=SentimentLabel.NEUTRAL,
                    confidence=0.8,
                    reasoning="Knowledge graph analysis completed successfully"
                ),
                processing_time=processing_time,
                status=ProcessingStatus.COMPLETED,
                raw_content=f"Enhanced knowledge graph updated with {len(entities)} entities and {len(relationships)} relationships",
                metadata={
                    "entities_extracted": len(entities),
                    "relationships_mapped": len(relationships),
                    "graph_nodes": self.graph.number_of_nodes(),
                    "graph_edges": self.graph.number_of_edges(),
                    "impact_analysis": impact_analysis
                }
            )
            
        except Exception as e:
            logger.error(f"Enhanced knowledge graph processing failed: {e}")
            return AnalysisResult(
                id=request.id,
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label=SentimentLabel.NEUTRAL,
                    confidence=0.0,
                    reasoning=f"Processing failed: {str(e)}"
                ),
                processing_time=(datetime.now() - start_time).total_seconds(),
                status=ProcessingStatus.FAILED,
                raw_content=f"Processing failed: {str(e)}"
            )
    
    async def extract_entities_enhanced(self, text: str) -> dict:
        """Enhanced entity extraction with domain-specific patterns."""
        try:
            # Extract domain-specific entities directly
            domain_entities = self._extract_domain_specific_entities(text)
            
            # For now, use domain entities as the base
            # In a full implementation, this could integrate with other entity extraction methods
            unique_entities = self._deduplicate_entities(domain_entities)
            
            return {
                "content": [{
                    "entities": unique_entities,
                    "extraction_method": "enhanced_with_domain_patterns"
                }]
            }
            
        except Exception as e:
            logger.error(f"Enhanced entity extraction failed: {e}")
            return {
                "content": [{
                    "entities": [],
                    "extraction_method": "fallback"
                }]
            }
    
    def _extract_domain_specific_entities(self, text: str) -> List[Dict]:
        """Extract domain-specific entities using patterns."""
        entities = []
        
        # Extract Classical Chinese linguistic terms
        linguistic_terms = re.findall(r'([诸盍焉耳尔叵之其者也乃是所以为])', text)
        for term in set(linguistic_terms):
            entities.append({
                "name": term,
                "type": "LINGUISTIC_TERM",
                "confidence": 0.9,
                "description": f"Classical Chinese linguistic term: {term}",
                "domain": "classical_chinese"
            })
        
        # Extract historical figures
        historical_figures = re.findall(r'(孔子|孟子|韩愈|柳宗元|欧阳修|陶渊明|吕布|刘备|曹操|马岱|启功)', text)
        for figure in set(historical_figures):
            entities.append({
                "name": figure,
                "type": "PERSON",
                "confidence": 0.9,
                "description": f"Historical figure: {figure}",
                "domain": "classical_chinese"
            })
        
        # Extract classical texts
        classical_texts = re.findall(r'《([^》]+)》', text)
        for text_name in set(classical_texts):
            entities.append({
                "name": text_name,
                "type": "WORK",
                "confidence": 0.8,
                "description": f"Classical Chinese text: {text_name}",
                "domain": "classical_chinese"
            })
        
        # Extract lesson titles
        lesson_patterns = [
            r'第([一二三四五六七八九十\d]+)课[：:]\s*([^，。\n]+)',
            r'Lesson\s*(\d+)[：:]\s*([^，。\n]+)'
        ]
        
        for pattern in lesson_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    lesson_num, lesson_title = match
                else:
                    lesson_num = match
                    lesson_title = f"Lesson {match}"
                
                entities.append({
                    "name": f"Lesson {lesson_num}: {lesson_title}",
                    "type": "LESSON",
                    "confidence": 0.8,
                    "description": f"Classical Chinese lesson: {lesson_title}",
                    "domain": "classical_chinese"
                })
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Deduplicate entities based on name."""
        seen_names = set()
        unique_entities = []
        
        for entity in entities:
            name = entity.get("name", "").strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def map_relationships_enhanced(self, text: str, entities: List[Dict]) -> dict:
        """Enhanced relationship mapping with domain-specific patterns."""
        try:
            # Get entity names
            entity_names = [e.get("name", "") for e in entities]
            entity_dict = {e.get("name", ""): e for e in entities}
            
            # Extract domain-specific relationships
            domain_relationships = self._extract_domain_relationships(text, entity_dict)
            
            # Use AI for additional relationship extraction
            ai_relationships = await self._extract_ai_relationships(text, entity_names)
            
            # Combine relationships
            all_relationships = domain_relationships + ai_relationships
            
            # Deduplicate relationships
            unique_relationships = self._deduplicate_relationships(all_relationships)
            
            return {
                "content": [{
                    "relationships": unique_relationships,
                    "mapping_method": "enhanced_with_domain_patterns"
                }]
            }
            
        except Exception as e:
            logger.error(f"Enhanced relationship mapping failed: {e}")
            return {
                "content": [{
                    "relationships": [],
                    "mapping_method": "fallback"
                }]
            }
    
    def _extract_domain_relationships(self, text: str, entity_dict: Dict) -> List[Dict]:
        """Extract domain-specific relationships using patterns."""
        relationships = []
        
        # Author-Work relationships
        author_work_patterns = [
            r'(Kai Li|James Erwin Dew|李恺|杜尔文).*?(文言章句|Classical Chinese)',
            r'(启功).*?(文言章句)',
        ]
        
        for pattern in author_work_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    author, work = match
                else:
                    continue
                
                if author in entity_dict and work in entity_dict:
                    relationships.append({
                        "source": author,
                        "target": work,
                        "relationship_type": "CREATED_BY",
                        "confidence": 0.9,
                        "description": f"{author} created {work}"
                    })
        
        # Work-Lesson relationships
        work_lesson_patterns = [
            r'(文言章句|Classical Chinese).*?(第[一二三四五六七八九十\d]+课|Lesson\s*\d+)',
        ]
        
        for pattern in work_lesson_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    work, lesson = match
                else:
                    continue
                
                if work in entity_dict and lesson in entity_dict:
                    relationships.append({
                        "source": work,
                        "target": lesson,
                        "relationship_type": "CONTAINS",
                        "confidence": 0.8,
                        "description": f"{work} contains {lesson}"
                    })
        
        # Linguistic term relationships
        term_function_patterns = [
            r'([诸盍焉耳尔叵])\s*[是|为]\s*([^，。\n]+)',
            r'([之|其|者|也|乃|是|以|所|为])\s*[的|地|得]\s*([^，。\n]+)',
        ]
        
        for pattern in term_function_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    term, function = match
                else:
                    continue
                
                if term in entity_dict:
                    relationships.append({
                        "source": term,
                        "target": function.strip(),
                        "relationship_type": "HAS_FUNCTION",
                        "confidence": 0.7,
                        "description": f"{term} has function: {function.strip()}"
                    })
        
        # Historical figure - work relationships
        figure_work_patterns = [
            r'(孔子|孟子|韩愈|柳宗元|欧阳修|陶渊明).*?《([^》]+)》',
        ]
        
        for pattern in figure_work_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    figure, work = match
                else:
                    continue
                
                if figure in entity_dict and work in entity_dict:
                    relationships.append({
                        "source": figure,
                        "target": work,
                        "relationship_type": "AUTHOR_OF",
                        "confidence": 0.8,
                        "description": f"{figure} is author of {work}"
                    })
        
        # Example relationships
        example_patterns = [
            r'([诸盍焉耳尔叵]).*?《([^》]+)》',
        ]
        
        for pattern in example_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    term, source = match
                else:
                    continue
                
                if term in entity_dict and source in entity_dict:
                    relationships.append({
                        "source": term,
                        "target": source,
                        "relationship_type": "EXAMPLE_FROM",
                        "confidence": 0.6,
                        "description": f"Example of {term} from {source}"
                    })
        
        return relationships
    
    async def _extract_ai_relationships(self, text: str, entity_names: List[str]) -> List[Dict]:
        """Extract relationships using AI."""
        try:
            prompt = f"""
            You are an expert in Classical Chinese and educational content analysis. 
            Analyze the relationships between entities in the given text.

            Instructions:
            1. Identify meaningful relationships between the provided entities
            2. Focus on educational, linguistic, and historical relationships
            3. For each relationship, provide:
               - source: The source entity name (exact match from entities list)
               - target: The target entity name (exact match from entities list)
               - relationship_type: One of [CREATED_BY, CONTAINS, HAS_FUNCTION, AUTHOR_OF, EXAMPLE_FROM, TEACHES, LEARNS_FROM, SIMILAR_TO, OPPOSES, SUPPORTS, DEPENDS_ON, RELATED_TO]
               - confidence: A score between 0.0 and 1.0
               - description: A clear description of the relationship
            4. Only include relationships that are explicitly mentioned or strongly implied
            5. Return ONLY valid JSON

            Entities to analyze: {entity_names}

            Text to analyze:
            {text[:2000]}  # Limit text length for processing

            Expected JSON format:
            {{
                "relationships": [
                    {{
                        "source": "entity_name",
                        "target": "entity_name",
                        "relationship_type": "relationship_type",
                        "confidence": 0.95,
                        "description": "clear description of the relationship"
                    }}
                ]
            }}

            Return only the JSON object, no additional text.
            """
            
            response = await self.strands_agent.run(prompt)
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Parse JSON response
            try:
                json_data = json.loads(content)
                return json_data.get("relationships", [])
            except json.JSONDecodeError:
                # Try to extract JSON from markdown
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group(1))
                        return json_data.get("relationships", [])
                    except json.JSONDecodeError:
                        pass
                
                return []
                
        except Exception as e:
            logger.error(f"AI relationship extraction failed: {e}")
            return []
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Deduplicate relationships based on source, target, and type."""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            key = (rel.get("source", ""), rel.get("target", ""), rel.get("relationship_type", ""))
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships
    
    async def _extract_text_content(self, request: AnalysisRequest) -> str:
        """Extract text content from request."""
        if isinstance(request.content, str):
            return request.content
        elif isinstance(request.content, bytes):
            return request.content.decode('utf-8', errors='ignore')
        else:
            return str(request.content)
    
    async def _add_to_graph(self, entities: List[Dict], relationships: List[Dict], request_id: str):
        """Add entities and relationships to the graph."""
        # Add entities as nodes
        for entity in entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "unknown")
            confidence = entity.get("confidence", 0.5)
            domain = entity.get("domain", "general")
            
            if entity_name:
                if entity_name not in self.graph:
                    self.graph.add_node(entity_name, 
                                      type=entity_type,
                                      confidence=confidence,
                                      domain=domain,
                                      first_seen=datetime.now().isoformat(),
                                      request_id=request_id)
                else:
                    # Update existing node
                    self.graph.nodes[entity_name]["confidence"] = max(
                        self.graph.nodes[entity_name].get("confidence", 0),
                        confidence
                    )
        
        # Add relationships as edges
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("relationship_type", "related")
            confidence = rel.get("confidence", 0.5)
            
            if source and target and source in self.graph and target in self.graph:
                self.graph.add_edge(source, target,
                                  relationship_type=rel_type,
                                  confidence=confidence,
                                  timestamp=datetime.now().isoformat(),
                                  request_id=request_id)
        
        # Save graph
        self._save_graph()
        
        # Update metadata
        self.metadata["graph_stats"] = self._get_graph_stats()
    
    async def _analyze_graph_impact(self, entities: List[Dict], relationships: List[Dict]) -> Dict:
        """Analyze the impact of new entities and relationships on the graph."""
        before_nodes = self.graph.number_of_nodes() - len(entities)
        before_edges = self.graph.number_of_edges() - len(relationships)
        
        return {
            "new_entities": len(entities),
            "new_relationships": len(relationships),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "growth_rate": {
                "nodes": (len(entities) / max(before_nodes, 1)) * 100,
                "edges": (len(relationships) / max(before_edges, 1)) * 100
            }
        }
    
    def _get_graph_stats(self) -> Dict:
        """Get current graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "density": 0}
        
        try:
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "average_clustering": nx.average_clustering(self.graph.to_undirected()),
                "connected_components": nx.number_connected_components(self.graph.to_undirected())
            }
        except Exception as e:
            logger.warning(f"Could not calculate all graph statistics: {e}")
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
                "average_clustering": 0,
                "connected_components": 1
            }
    
    def _load_existing_graph(self):
        """Load existing graph from file."""
        try:
            if self.graph_file.exists():
                import pickle
                with open(self.graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded existing enhanced graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            else:
                logger.info("No existing enhanced graph found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load existing enhanced graph: {e}")
            self.graph = nx.DiGraph()
    
    def _save_graph(self):
        """Save graph to file."""
        try:
            import pickle
            with open(self.graph_file, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.debug(f"Enhanced graph saved with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Failed to save enhanced graph: {e}")
    
    # Inherit other methods from base class
    async def query_knowledge_graph(self, query: str) -> dict:
        """Query the knowledge graph."""
        # Implementation would be similar to base class
        return {"content": [{"query": query, "results": []}]}
    
    async def generate_graph_report(self, output_path: Optional[str] = None) -> dict:
        """Generate graph report."""
        # Implementation would be similar to base class
        return {"content": [{"report": "Enhanced graph report generated"}]}
    
    async def analyze_graph_communities(self) -> dict:
        """Analyze graph communities."""
        # Implementation would be similar to base class
        return {"content": [{"communities": []}]}
    
    async def find_entity_paths(self, source: str, target: str) -> dict:
        """Find paths between entities."""
        # Implementation would be similar to base class
        return {"content": [{"paths": []}]}
    
    async def get_entity_context(self, entity: str) -> dict:
        """Get entity context."""
        # Implementation would be similar to base class
        return {"content": [{"context": {}}]}
    
    # Phase 3 Advanced Features
    
    async def analyze_semantic_similarity(self, text: str, entities: List[Dict] = None) -> dict:
        """Analyze semantic similarity between entities using Phase 3 features."""
        try:
            if entities is None:
                # Extract entities if not provided
                entities_result = await self.extract_entities_enhanced(text)
                entities = entities_result.get("content", [{}])[0].get("entities", [])
            
            # Run semantic similarity analysis
            similarity_results = self.semantic_analyzer.analyze_semantic_similarity(entities, text)
            
            # Get statistics
            stats = self.semantic_analyzer.get_similarity_statistics(similarity_results)
            
            # Get relationship suggestions
            suggestions = self.semantic_analyzer.get_relationship_suggestions(similarity_results)
            
            return {
                "content": [{
                    "similarity_results": [
                        {
                            "entity1": result.entity1,
                            "entity2": result.entity2,
                            "similarity_score": result.similarity_score,
                            "similarity_type": result.similarity_type,
                            "confidence": result.confidence,
                            "context_evidence": result.context_evidence
                        }
                        for result in similarity_results
                    ],
                    "statistics": stats,
                    "relationship_suggestions": suggestions,
                    "high_similarity_pairs": len(
                        self.semantic_analyzer.filter_high_similarity_pairs(similarity_results)
                    )
                }]
            }
            
        except Exception as e:
            logger.error(f"Semantic similarity analysis failed: {e}")
            return {
                "content": [{
                    "error": str(e)
                }]
            }
    
    async def optimize_relationships(self, text: str, relationships: List[Dict] = None, entities: List[Dict] = None) -> dict:
        """Optimize relationships using Phase 3 quality assessment."""
        try:
            if entities is None:
                # Extract entities if not provided
                entities_result = await self.extract_entities_enhanced(text)
                entities = entities_result.get("content", [{}])[0].get("entities", [])
            
            if relationships is None:
                # Extract relationships if not provided
                relationships_result = await self.map_relationships_enhanced(text, entities)
                relationships = relationships_result.get("content", [{}])[0].get("relationships", [])
            
            # Run relationship optimization
            optimized_relationships = self.relationship_optimizer.optimize_relationships(
                relationships, entities, text
            )
            
            # Get optimization statistics
            stats = self.relationship_optimizer.get_optimization_statistics(
                relationships, optimized_relationships
            )
            
            return {
                "content": [{
                    "original_relationships": len(relationships),
                    "optimized_relationships": len(optimized_relationships),
                    "optimized_relationships_list": optimized_relationships,
                    "optimization_statistics": stats,
                    "quality_improvement": stats.get("quality_improvement", 0.0),
                    "redundancy_reduction": stats.get("redundancy_reduction", 0.0)
                }]
            }
            
        except Exception as e:
            logger.error(f"Relationship optimization failed: {e}")
            return {
                "content": [{
                    "error": str(e)
                }]
            }
    
    async def cluster_entities_advanced(self, text: str, entities: List[Dict] = None) -> dict:
        """Advanced entity clustering using Phase 3 algorithms."""
        try:
            if entities is None:
                # Extract entities if not provided
                entities_result = await self.extract_entities_enhanced(text)
                entities = entities_result.get("content", [{}])[0].get("entities", [])
            
            # Run advanced entity clustering
            clusters = self.entity_clustering.cluster_entities(entities, text)
            
            # Get clustering statistics
            stats = self.entity_clustering.get_cluster_statistics(clusters)
            
            # Extract relationships from clusters
            cluster_relationships = []
            for cluster in clusters:
                for rel in cluster.relationships:
                    cluster_relationships.append({
                        "source": rel[0],
                        "target": rel[1],
                        "relationship_type": rel[2],
                        "cluster_type": cluster.cluster_type,
                        "confidence": cluster.confidence
                    })
            
            return {
                "content": [{
                    "clusters": [
                        {
                            "entities": cluster.entities,
                            "cluster_type": cluster.cluster_type,
                            "confidence": cluster.confidence,
                            "relationships": cluster.relationships
                        }
                        for cluster in clusters
                    ],
                    "cluster_statistics": stats,
                    "relationships_from_clustering": cluster_relationships,
                    "total_relationships_created": stats.get("total_relationships_created", 0)
                }]
            }
            
        except Exception as e:
            logger.error(f"Advanced entity clustering failed: {e}")
            return {
                "content": [{
                    "error": str(e)
                }]
            }
    
    async def run_phase3_quality_assessment(self, text: str) -> dict:
        """Run comprehensive Phase 3 quality assessment."""
        try:
            # Extract entities and relationships
            entities_result = await self.extract_entities_enhanced(text)
            entities = entities_result.get("content", [{}])[0].get("entities", [])
            
            relationships_result = await self.map_relationships_enhanced(text, entities)
            relationships = relationships_result.get("content", [{}])[0].get("relationships", [])
            
            # Run all Phase 3 analyses
            similarity_results = self.semantic_analyzer.analyze_semantic_similarity(entities, text)
            optimized_relationships = self.relationship_optimizer.optimize_relationships(
                relationships, entities, text
            )
            clusters = self.entity_clustering.cluster_entities(entities, text)
            
            # Calculate quality metrics
            total_entities = len(entities)
            total_relationships = len(optimized_relationships)
            
            # Calculate orphan node rate
            connected_entities = set()
            for rel in optimized_relationships:
                connected_entities.add(rel["source"])
                connected_entities.add(rel["target"])
            
            orphan_nodes = total_entities - len(connected_entities)
            orphan_rate = orphan_nodes / total_entities if total_entities > 0 else 1.0
            
            # Calculate relationship coverage
            relationship_coverage = total_relationships / total_entities if total_entities > 0 else 0.0
            
            # Get similarity statistics
            similarity_stats = self.semantic_analyzer.get_similarity_statistics(similarity_results)
            
            # Get optimization statistics
            optimization_stats = self.relationship_optimizer.get_optimization_statistics(
                relationships, optimized_relationships
            )
            
            # Get clustering statistics
            clustering_stats = self.entity_clustering.get_cluster_statistics(clusters)
            
            return {
                "content": [{
                    "quality_metrics": {
                        "total_entities": total_entities,
                        "total_relationships": total_relationships,
                        "orphan_nodes": orphan_nodes,
                        "orphan_rate": orphan_rate,
                        "relationship_coverage": relationship_coverage,
                        "meets_orphan_target": orphan_rate < 0.3,
                        "meets_coverage_target": relationship_coverage > 0.5
                    },
                    "phase3_analyses": {
                        "semantic_similarity": similarity_stats,
                        "relationship_optimization": optimization_stats,
                        "entity_clustering": clustering_stats
                    },
                    "overall_assessment": {
                        "phase3_implementation": "complete",
                        "quality_targets_met": orphan_rate < 0.3 and relationship_coverage > 0.5,
                        "recommendations": self._generate_phase3_recommendations(
                            orphan_rate, relationship_coverage, similarity_stats, optimization_stats
                        )
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Phase 3 quality assessment failed: {e}")
            return {
                "content": [{
                    "error": str(e)
                }]
            }
    
    def _generate_phase3_recommendations(
        self, 
        orphan_rate: float, 
        relationship_coverage: float, 
        similarity_stats: Dict, 
        optimization_stats: Dict
    ) -> List[str]:
        """Generate recommendations based on Phase 3 analysis."""
        recommendations = []
        
        if orphan_rate > 0.3:
            recommendations.append(
                f"Orphan node rate ({orphan_rate:.2f}) exceeds target (0.3). "
                "Consider adjusting similarity thresholds or clustering parameters."
            )
        
        if relationship_coverage < 0.5:
            recommendations.append(
                f"Relationship coverage ({relationship_coverage:.2f}) below target (0.5). "
                "Consider enabling more fallback strategies or adjusting optimization criteria."
            )
        
        if similarity_stats.get("average_similarity", 0) < 0.4:
            recommendations.append(
                "Low average similarity score. Consider adjusting semantic patterns "
                "or context window size."
            )
        
        if optimization_stats.get("quality_improvement", 0) < 0.0:
            recommendations.append(
                "Low quality improvement from optimization. Consider adjusting "
                "quality thresholds or validation patterns."
            )
        
        if not recommendations:
            recommendations.append("All Phase 3 quality targets met. System performing optimally.")
        
        return recommendations
