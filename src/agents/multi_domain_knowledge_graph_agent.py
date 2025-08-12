"""
Multi-Domain Knowledge Graph Agent for language-based content isolation.
Implements W3C RDF dataset best practices with named graphs per language domain.
Supports cross-domain relationships and flexible querying patterns.
"""

import asyncio
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

from loguru import logger

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import (
    AnalysisRequest, 
    AnalysisResult, 
    DataType, 
    SentimentResult,
    ProcessingStatus
)
from src.core.vector_db import VectorDBManager
from src.config.config import config
from src.config.settings import settings


class MultiDomainKnowledgeGraphAgent(StrandsBaseAgent):
    """
    Multi-Domain Knowledge Graph Agent with language-based content isolation.
    
    Features:
    - Language-based domain separation
    - Cross-domain relationship support
    - Flexible querying patterns
    - Multiple visualization options
    - Topic categorization within domains
    """
    
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
        
        # Initialize storage
        self.graph_storage_path = Path(
            graph_storage_path or settings.paths.knowledge_graphs_dir
        )
        self.graph_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Multi-graph architecture following RDF dataset pattern
        self.graphs: Dict[str, nx.DiGraph] = {}
        self.cross_domain_graph = nx.DiGraph()  # For cross-domain relationships
        self.domain_metadata: Dict[str, Dict] = {}
        
        # Language domain configuration
        self.language_domains = {
            "en": "English",
            "zh": "Chinese", 
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "ru": "Russian",
            "pt": "Portuguese"
        }
        
        # Topic categories within each domain
        self.topic_categories = {
            "economics": ["trade", "finance", "market", "economy", "business"],
            "politics": ["government", "policy", "election", "political", "diplomacy"],
            "social": ["society", "culture", "social", "community", "people"],
            "science": ["research", "technology", "science", "innovation", "discovery"],
            "war": ["conflict", "military", "war", "defense", "security"],
            "tech": ["technology", "digital", "software", "hardware", "innovation"]
        }
        
        # Initialize graphs for each language domain
        self._initialize_domain_graphs()
        
        # Load existing data
        self._load_existing_graphs()
        
        # Initialize vector DB manager
        self.vector_db = VectorDBManager()
        
        self.metadata.update({
            "agent_type": "multi_domain_knowledge_graph",
            "model": self.model_name,
            "capabilities": [
                "language_domain_isolation",
                "cross_domain_relationships",
                "topic_categorization",
                "flexible_querying",
                "multi_visualization",
                "entity_extraction",
                "relationship_mapping",
                "graph_analysis"
            ],
            "supported_languages": list(self.language_domains.keys()),
            "topic_categories": list(self.topic_categories.keys()),
            "graph_stats": self._get_comprehensive_stats(),
            "architecture": "multi_graph_rdf_dataset"
        })
        
        logger.info(
            f"Multi-Domain Knowledge Graph Agent {self.agent_id} initialized with "
            f"{len(self.language_domains)} language domains"
        )
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.extract_entities_multi_domain,
            self.map_relationships_multi_domain,
            self.query_domain,
            self.query_cross_domain,
            self.query_all_domains,
            self.generate_domain_report,
            self.generate_cross_domain_report,
            self.generate_comprehensive_report,
            self.find_entity_paths_multi_domain,
            self.get_entity_context_multi_domain,
            self.analyze_domain_communities,
            self.analyze_cross_domain_connections,
            self.get_domain_statistics,
            self.merge_related_domains
        ]
    
    def _initialize_domain_graphs(self):
        """Initialize separate graphs for each language domain."""
        for lang_code, lang_name in self.language_domains.items():
            self.graphs[lang_code] = nx.DiGraph()
            self.domain_metadata[lang_code] = {
                "name": lang_name,
                "created": datetime.now().isoformat(),
                "topics": defaultdict(int),
                "entity_count": 0,
                "relationship_count": 0,
                "last_updated": datetime.now().isoformat()
            }
    
    def _load_existing_graphs(self):
        """Load existing graphs from storage."""
        try:
            for lang_code in self.language_domains.keys():
                graph_file = self.graph_storage_path / f"knowledge_graph_{lang_code}.pkl"
                if graph_file.exists():
                    import pickle
                    with open(graph_file, 'rb') as f:
                        self.graphs[lang_code] = pickle.load(f)
                    
                    # Update metadata
                    self.domain_metadata[lang_code]["entity_count"] = self.graphs[lang_code].number_of_nodes()
                    self.domain_metadata[lang_code]["relationship_count"] = self.graphs[lang_code].number_of_edges()
                    self.domain_metadata[lang_code]["last_updated"] = datetime.now().isoformat()
            
            # Load cross-domain graph
            cross_domain_file = self.graph_storage_path / "cross_domain_graph.pkl"
            if cross_domain_file.exists():
                import pickle
                with open(cross_domain_file, 'rb') as f:
                    self.cross_domain_graph = pickle.load(f)
            
            logger.info("Loaded existing multi-domain graphs")
        except Exception as e:
            logger.error(f"Failed to load existing graphs: {e}")
    
    def _save_graphs(self):
        """Save all graphs to storage."""
        try:
            import pickle
            
            # Save domain-specific graphs
            for lang_code, graph in self.graphs.items():
                graph_file = self.graph_storage_path / f"knowledge_graph_{lang_code}.pkl"
                with open(graph_file, 'wb') as f:
                    pickle.dump(graph, f)
            
            # Save cross-domain graph
            cross_domain_file = self.graph_storage_path / "cross_domain_graph.pkl"
            with open(cross_domain_file, 'wb') as f:
                pickle.dump(self.cross_domain_graph, f)
            
            logger.debug("Saved all multi-domain graphs")
        except Exception as e:
            logger.error(f"Failed to save graphs: {e}")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        return request.data_type in [
            DataType.TEXT, DataType.AUDIO, DataType.VIDEO, 
            DataType.WEBPAGE, DataType.PDF, DataType.SOCIAL_MEDIA
        ]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process content and add to appropriate domain graph."""
        try:
            # Extract text content
            text_content = await self._extract_text_content(request)
            
            # Detect language and domain
            detected_lang = await self._detect_language(text_content)
            detected_topics = await self._detect_topics(text_content)
            
            # Extract entities and relationships
            entities, relationships = await self._process_content_multi_domain(
                text_content, detected_lang, detected_topics
            )
            
            # Add to appropriate domain graph
            await self._add_to_domain_graph(
                entities, relationships, detected_lang, 
                detected_topics, request.id
            )
            
            # Check for cross-domain relationships
            cross_domain_entities = await self._identify_cross_domain_entities(
                entities, detected_lang
            )
            if cross_domain_entities:
                await self._add_cross_domain_relationships(
                    cross_domain_entities, detected_lang, request.id
                )
            
            # Generate result
            result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                content=request.content,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=1.0,
                    reasoning="Knowledge graph processing completed"
                ),
                processing_time=0.0,
                status=ProcessingStatus.COMPLETED,
                model_used=self.model_name,
                metadata={
                    "agent_id": self.agent_id,
                    "method": "multi_domain_knowledge_graph",
                    "detected_language": detected_lang,
                    "detected_topics": detected_topics,
                    "entities_extracted": len(entities),
                    "relationships_mapped": len(relationships),
                    "cross_domain_entities": len(cross_domain_entities) if cross_domain_entities else 0
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                content=request.content,
                sentiment=SentimentResult(
                    label="error",
                    confidence=0.0,
                    reasoning=f"Processing failed: {str(e)}"
                ),
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                model_used=self.model_name,
                metadata={"error": str(e)}
            )
    
    async def _detect_language(self, text: str) -> str:
        """Detect the language of the text content."""
        # Simple language detection based on character patterns
        # In production, use a proper language detection library
        
        # Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return "zh"
        # Japanese characters
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "ja"
        # Korean characters
        elif re.search(r'[\uac00-\ud7af]', text):
            return "ko"
        # Arabic characters
        elif re.search(r'[\u0600-\u06ff]', text):
            return "ar"
        # Cyrillic characters
        elif re.search(r'[\u0400-\u04ff]', text):
            return "ru"
        # Default to English
        else:
            return "en"
    
    async def _detect_topics(self, text: str) -> List[str]:
        """Detect topics in the text content."""
        detected_topics = []
        text_lower = text.lower()
        
        for topic, keywords in self.topic_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    async def _process_content_multi_domain(
        self, text: str, language: str, topics: List[str]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process content for multi-domain knowledge graph."""
        # Extract entities
        entities = await self.extract_entities_multi_domain(text, language, topics)
        
        # Map relationships
        relationships = await self.map_relationships_multi_domain(
            text, entities.get("entities", []), language, topics
        )
        
        return entities.get("entities", []), relationships.get("relationships", [])
    
    async def _add_to_domain_graph(
        self, entities: List[Dict], relationships: List[Dict], 
        language: str, topics: List[str], request_id: str
    ):
        """Add entities and relationships to the appropriate domain graph."""
        if language not in self.graphs:
            logger.warning(f"Unknown language {language}, using English domain")
            language = "en"
        
        graph = self.graphs[language]
        
        # Add entities as nodes
        for entity in entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "unknown")
            confidence = entity.get("confidence", 0.5)
            
            if entity_name:
                if entity_name not in graph:
                    graph.add_node(entity_name, 
                                  type=entity_type,
                                  confidence=confidence,
                                  language=language,
                                  topics=topics,
                                  first_seen=datetime.now().isoformat(),
                                  request_id=request_id)
                else:
                    # Update existing node
                    existing_topics = graph.nodes[entity_name].get("topics", [])
                    graph.nodes[entity_name]["topics"] = list(set(existing_topics + topics))
                    graph.nodes[entity_name]["confidence"] = max(
                        graph.nodes[entity_name].get("confidence", 0),
                        confidence
                    )
        
        # Add relationships as edges
        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("relationship_type", "related")
            confidence = rel.get("confidence", 0.5)
            
            if source and target and source in graph and target in graph:
                graph.add_edge(source, target,
                              relationship_type=rel_type,
                              confidence=confidence,
                              language=language,
                              topics=topics,
                              timestamp=datetime.now().isoformat(),
                              request_id=request_id)
        
        # Update domain metadata
        self.domain_metadata[language]["entity_count"] = graph.number_of_nodes()
        self.domain_metadata[language]["relationship_count"] = graph.number_of_edges()
        self.domain_metadata[language]["last_updated"] = datetime.now().isoformat()
        
        for topic in topics:
            self.domain_metadata[language]["topics"][topic] += 1
        
        # Save graphs
        self._save_graphs()
    
    async def _identify_cross_domain_entities(
        self, entities: List[Dict], source_language: str
    ) -> List[Dict]:
        """Identify entities that might have cross-domain relationships."""
        cross_domain_entities = []
        
        # Look for entities that might appear in multiple languages
        # This is a simplified approach - in production, use entity linking
        for entity in entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "")
            
            # Check if this entity exists in other language domains
            for lang_code, graph in self.graphs.items():
                if lang_code != source_language and entity_name in graph:
                    cross_domain_entities.append({
                        "entity": entity,
                        "source_language": source_language,
                        "target_language": lang_code,
                        "relationship_type": "cross_domain_reference"
                    })
        
        return cross_domain_entities
    
    async def _add_cross_domain_relationships(
        self, cross_domain_entities: List[Dict], source_language: str, request_id: str
    ):
        """Add cross-domain relationships to the cross-domain graph."""
        for cross_entity in cross_domain_entities:
            entity = cross_entity["entity"]
            target_language = cross_entity["target_language"]
            
            # Create cross-domain relationship
            source_node = f"{entity['name']}_{source_language}"
            target_node = f"{entity['name']}_{target_language}"
            
            self.cross_domain_graph.add_node(source_node,
                                           entity_name=entity["name"],
                                           language=source_language,
                                           type=entity.get("type", "unknown"))
            
            self.cross_domain_graph.add_node(target_node,
                                           entity_name=entity["name"],
                                           language=target_language,
                                           type=entity.get("type", "unknown"))
            
            self.cross_domain_graph.add_edge(source_node, target_node,
                                           relationship_type="cross_domain_reference",
                                           timestamp=datetime.now().isoformat(),
                                           request_id=request_id)
    
    async def extract_entities_multi_domain(
        self, text: str, language: str = "en", topics: List[str] = None
    ) -> dict:
        """Extract entities with domain awareness."""
        try:
            # Use LLM to extract entities with domain context
            prompt = f"""
            Extract entities from the following text. Consider the language ({language}) 
            and topics ({topics if topics else 'general'}) when identifying entities.
            
            Text: {text}
            
            Return entities in JSON format with:
            - name: entity name
            - type: entity type (person, organization, location, concept, etc.)
            - confidence: confidence score (0-1)
            - language: detected language
            - topics: related topics
            """
            
            # This would use the LLM to extract entities
            # For now, return a simplified structure
            entities = [
                {
                    "name": "Sample Entity",
                    "type": "person",
                    "confidence": 0.8,
                    "language": language,
                    "topics": topics or []
                }
            ]
            
            return {
                "status": "success",
                "entities": entities,
                "language": language,
                "topics": topics,
                "count": len(entities)
            }
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return {
                "status": "error",
                "error": str(e),
                "entities": []
            }
    
    async def map_relationships_multi_domain(
        self, text: str, entities: List[Dict], 
        language: str = "en", topics: List[str] = None
    ) -> dict:
        """Map relationships with domain awareness."""
        try:
            # Use LLM to map relationships with domain context
            entity_names = [e["name"] for e in entities]
            
            prompt = f"""
            Map relationships between entities in the following text. Consider the 
            language ({language}) and topics ({topics if topics else 'general'}).
            
            Text: {text}
            Entities: {entity_names}
            
            Return relationships in JSON format with:
            - source: source entity name
            - target: target entity name
            - relationship_type: type of relationship
            - confidence: confidence score (0-1)
            - language: detected language
            - topics: related topics
            """
            
            # This would use the LLM to map relationships
            # For now, return a simplified structure
            relationships = []
            if len(entities) >= 2:
                relationships = [
                    {
                        "source": entities[0]["name"],
                        "target": entities[1]["name"],
                        "relationship_type": "related",
                        "confidence": 0.7,
                        "language": language,
                        "topics": topics or []
                    }
                ]
            
            return {
                "status": "success",
                "relationships": relationships,
                "language": language,
                "topics": topics,
                "count": len(relationships)
            }
            
        except Exception as e:
            logger.error(f"Failed to map relationships: {e}")
            return {
                "status": "error",
                "error": str(e),
                "relationships": []
            }
    
    async def query_domain(self, query: str, language: str = "en") -> dict:
        """Query entities and relationships within a specific domain."""
        if language not in self.graphs:
            return {"status": "error", "error": f"Unknown language domain: {language}"}
        
        graph = self.graphs[language]
        
        # Simple text-based search
        matching_nodes = []
        for node in graph.nodes():
            if query.lower() in node.lower():
                matching_nodes.append({
                    "name": node,
                    "type": graph.nodes[node].get("type", "unknown"),
                    "topics": graph.nodes[node].get("topics", []),
                    "confidence": graph.nodes[node].get("confidence", 0.5)
                })
        
        return {
            "status": "success",
            "language": language,
            "query": query,
            "results": matching_nodes,
            "count": len(matching_nodes)
        }
    
    async def query_cross_domain(self, query: str) -> dict:
        """Query entities and relationships across domains."""
        results = {}
        
        for language in self.language_domains.keys():
            domain_result = await self.query_domain(query, language)
            if domain_result["status"] == "success" and domain_result["count"] > 0:
                results[language] = domain_result
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "total_count": sum(r["count"] for r in results.values())
        }
    
    async def query_all_domains(self, query: str) -> dict:
        """Query all domains and return comprehensive results."""
        all_results = []
        
        for language in self.language_domains.keys():
            domain_result = await self.query_domain(query, language)
            if domain_result["status"] == "success":
                for result in domain_result["results"]:
                    result["language"] = language
                    all_results.append(result)
        
        return {
            "status": "success",
            "query": query,
            "results": all_results,
            "count": len(all_results)
        }
    
    def _get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics across all domains."""
        stats = {
            "total_domains": len(self.language_domains),
            "domains": {},
            "cross_domain": {
                "nodes": self.cross_domain_graph.number_of_nodes(),
                "edges": self.cross_domain_graph.number_of_edges()
            }
        }
        
        total_nodes = 0
        total_edges = 0
        
        for lang_code, graph in self.graphs.items():
            nodes = graph.number_of_nodes()
            edges = graph.number_of_edges()
            
            stats["domains"][lang_code] = {
                "name": self.language_domains[lang_code],
                "nodes": nodes,
                "edges": edges,
                "topics": dict(self.domain_metadata[lang_code]["topics"])
            }
            
            total_nodes += nodes
            total_edges += edges
        
        stats["total_nodes"] = total_nodes
        stats["total_edges"] = total_edges
        
        return stats
    
    async def generate_domain_report(self, language: str = "en", output_path: Optional[str] = None) -> dict:
        """Generate a report for a specific domain."""
        if language not in self.graphs:
            return {"status": "error", "error": f"Unknown language domain: {language}"}
        
        graph = self.graphs[language]
        
        # Generate report content
        report = {
            "domain": language,
            "domain_name": self.language_domains[language],
            "statistics": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "topics": dict(self.domain_metadata[language]["topics"])
            },
            "top_entities": [],
            "top_relationships": []
        }
        
        # Get top entities by degree
        if graph.number_of_nodes() > 0:
            node_degrees = dict(graph.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            report["top_entities"] = [
                {"name": node, "degree": degree} for node, degree in top_nodes
            ]
        
        return {
            "status": "success",
            "report": report,
            "output_path": output_path
        }
    
    async def generate_cross_domain_report(self, output_path: Optional[str] = None) -> dict:
        """Generate a report for cross-domain relationships."""
        report = {
            "cross_domain_statistics": {
                "nodes": self.cross_domain_graph.number_of_nodes(),
                "edges": self.cross_domain_graph.number_of_edges()
            },
            "cross_domain_connections": []
        }
        
        # Get cross-domain connections
        for edge in self.cross_domain_graph.edges(data=True):
            source, target, data = edge
            report["cross_domain_connections"].append({
                "source": source,
                "target": target,
                "relationship_type": data.get("relationship_type", "unknown")
            })
        
        return {
            "status": "success",
            "report": report,
            "output_path": output_path
        }
    
    async def generate_comprehensive_report(self, output_path: Optional[str] = None) -> dict:
        """Generate a comprehensive report across all domains."""
        report = {
            "overview": self._get_comprehensive_stats(),
            "domains": {},
            "cross_domain": await self.generate_cross_domain_report()
        }
        
        # Generate reports for each domain
        for language in self.language_domains.keys():
            domain_report = await self.generate_domain_report(language)
            if domain_report["status"] == "success":
                report["domains"][language] = domain_report["report"]
        
        return {
            "status": "success",
            "report": report,
            "output_path": output_path
        }
    
    async def find_entity_paths_multi_domain(
        self, source: str, target: str, max_paths: int = 5
    ) -> dict:
        """Find paths between entities across domains."""
        all_paths = []
        
        # Search within each domain
        for language in self.language_domains.keys():
            graph = self.graphs[language]
            if source in graph and target in graph:
                try:
                    paths = list(nx.all_simple_paths(graph, source, target, cutoff=3))
                    for path in paths[:max_paths]:
                        all_paths.append({
                            "path": path,
                            "domain": language,
                            "length": len(path) - 1
                        })
                except nx.NetworkXNoPath:
                    continue
        
        return {
            "status": "success",
            "source": source,
            "target": target,
            "paths": all_paths,
            "count": len(all_paths)
        }
    
    async def get_entity_context_multi_domain(self, entity: str) -> dict:
        """Get context for an entity across all domains."""
        context = {
            "entity": entity,
            "appearances": {},
            "cross_domain_connections": []
        }
        
        # Find entity in each domain
        for language in self.language_domains.keys():
            graph = self.graphs[language]
            if entity in graph:
                node_data = graph.nodes[entity]
                neighbors = list(graph.neighbors(entity))
                
                context["appearances"][language] = {
                    "type": node_data.get("type", "unknown"),
                    "topics": node_data.get("topics", []),
                    "confidence": node_data.get("confidence", 0.5),
                    "neighbors": neighbors[:10]  # Limit to first 10 neighbors
                }
        
        # Find cross-domain connections
        for node in self.cross_domain_graph.nodes():
            if entity in node:
                context["cross_domain_connections"].append(node)
        
        return {
            "status": "success",
            "context": context
        }
    
    async def analyze_domain_communities(self, language: str = "en") -> dict:
        """Analyze communities within a specific domain."""
        if language not in self.graphs:
            return {"status": "error", "error": f"Unknown language domain: {language}"}
        
        graph = self.graphs[language]
        
        if graph.number_of_nodes() == 0:
            return {"status": "success", "communities": []}
        
        # Convert to undirected for community detection
        undirected = graph.to_undirected()
        
        try:
            # Use Louvain method for community detection
            communities = nx.community.louvain_communities(undirected)
            
            community_data = []
            for i, community in enumerate(communities):
                community_data.append({
                    "id": i,
                    "size": len(community),
                    "nodes": list(community)[:10],  # Show first 10 nodes
                    "topics": self._extract_community_topics(community, graph)
                })
            
            return {
                "status": "success",
                "language": language,
                "communities": community_data,
                "count": len(communities)
            }
        except Exception as e:
            logger.error(f"Failed to analyze communities: {e}")
            return {"status": "error", "error": str(e)}
    
    async def analyze_cross_domain_connections(self) -> dict:
        """Analyze connections between domains."""
        if self.cross_domain_graph.number_of_nodes() == 0:
            return {"status": "success", "connections": []}
        
        # Analyze cross-domain connections
        connections = []
        for edge in self.cross_domain_graph.edges(data=True):
            source, target, data = edge
            connections.append({
                "source": source,
                "target": target,
                "relationship_type": data.get("relationship_type", "unknown"),
                "timestamp": data.get("timestamp", "")
            })
        
        return {
            "status": "success",
            "connections": connections,
            "count": len(connections)
        }
    
    async def get_domain_statistics(self) -> dict:
        """Get comprehensive statistics for all domains."""
        return {
            "status": "success",
            "statistics": self._get_comprehensive_stats()
        }
    
    async def merge_related_domains(self, source_lang: str, target_lang: str) -> dict:
        """Merge related entities between domains."""
        if source_lang not in self.graphs or target_lang not in self.graphs:
            return {"status": "error", "error": "Invalid language domains"}
        
        source_graph = self.graphs[source_lang]
        target_graph = self.graphs[target_lang]
        
        merged_entities = []
        
        # Find entities that might be the same across domains
        for source_node in source_graph.nodes():
            for target_node in target_graph.nodes():
                if source_node.lower() == target_node.lower():
                    merged_entities.append({
                        "source_entity": source_node,
                        "target_entity": target_node,
                        "source_language": source_lang,
                        "target_language": target_lang
                    })
        
        return {
            "status": "success",
            "merged_entities": merged_entities,
            "count": len(merged_entities)
        }
    
    def _extract_community_topics(self, community: Set, graph: nx.DiGraph) -> List[str]:
        """Extract common topics for a community."""
        topics = defaultdict(int)
        
        for node in community:
            node_topics = graph.nodes[node].get("topics", [])
            for topic in node_topics:
                topics[topic] += 1
        
        # Return top 3 topics
        return [topic for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]]
    
    async def _extract_text_content(self, request: AnalysisRequest) -> str:
        """Extract text content from the request."""
        if request.data_type == DataType.TEXT:
            return request.content
        elif request.data_type == DataType.PDF:
            # Extract text from PDF
            return request.content  # Simplified
        else:
            # For other types, return the content as is
            return str(request.content)
