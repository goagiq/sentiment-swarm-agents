"""
Knowledge Graph Agent for entity extraction, relationship mapping, and graph analysis.
"""

import asyncio
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

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


class KnowledgeGraphAgent(StrandsBaseAgent):
    """Knowledge Graph Agent for entity extraction and relationship mapping."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        graph_storage_path: Optional[str] = None,
        **kwargs
    ):
        # Set model name before calling super().__init__
        self.model_name = model_name or config.model.default_text_model
        
        super().__init__(
            model_name=self.model_name,
            **kwargs
        )
        
        # Initialize graph storage - use settings
        self.graph_storage_path = Path(
            graph_storage_path or settings.paths.knowledge_graphs_dir
        )
        self.graph_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize NetworkX graph
        self.graph = nx.DiGraph()
        self.graph_file = self.graph_storage_path / "knowledge_graph.pkl"
        self._load_existing_graph()
        
        # Initialize vector DB manager
        self.vector_db = VectorDBManager()
        
        # Agent metadata with model name properly set
        self.metadata.update({
            "agent_type": "knowledge_graph",
            "model": self.model_name,  # Fix: Set model name in metadata
            "capabilities": [
                "entity_extraction",
                "relationship_mapping", 
                "graph_analysis",
                "graph_visualization",
                "knowledge_inference",
                "community_detection",
                "chunk_based_processing"
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
            "chunk_size": 1200,  # GraphRAG-inspired chunk size
            "chunk_overlap": 100  # GraphRAG-inspired overlap
        })
        
        logger.info(
            f"Knowledge Graph Agent {self.agent_id} initialized with model "
            f"{self.model_name}"
        )
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.extract_entities,
            self.map_relationships,
            self.query_knowledge_graph,
            self.generate_graph_report,
            self.analyze_graph_communities,
            self.find_entity_paths,
            self.get_entity_context
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type in self.metadata["supported_data_types"]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process content and build knowledge graph using GraphRAG-inspired approach."""
        try:
            # Extract text content from various data types
            text_content = await self._extract_text_content(request)
            
            # Use chunk-based processing for better scalability
            if len(text_content) > 2000:  # Use chunks for longer texts
                logger.info("Using chunk-based processing for large text")
                entities, relationships = await self._process_text_chunks(text_content)
            else:
                # For shorter texts, use direct processing with combined extraction
                logger.info("Using direct processing with combined entity and relationship extraction")
                extraction_result = await self.extract_entities(text_content)
                json_data = extraction_result.get("content", [{}])[0].get("json", {})
                entities = json_data.get("entities", [])
                relationships = json_data.get("relationships", [])
            
            # Add entities and relationships to the graph
            await self._add_to_graph(entities, relationships, request.id)
            
            # Generate graph analysis
            graph_analysis = await self._analyze_graph_impact(entities, relationships)
            
            # Create analysis result
            result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",  # Knowledge graph doesn't do sentiment
                    confidence=1.0,
                    reasoning="Knowledge graph analysis completed successfully using GraphRAG-inspired approach"
                ),
                processing_time=0.0,  # Will be set by base class
                status=ProcessingStatus.COMPLETED,
                raw_content=str(request.content),
                extracted_text=text_content,
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.metadata.get("model", "unknown"),  # Safe access
                    "entities_extracted": len(entities),
                    "relationships_mapped": len(relationships),
                    "graph_nodes": self.graph.number_of_nodes(),
                    "graph_edges": self.graph.number_of_edges(),
                    "graph_analysis": graph_analysis,
                    "method": "graphrag_inspired_knowledge_graph_analysis",
                    "processing_approach": "chunk_based" if len(text_content) > 2000 else "direct"
                }
            )
            
            # Store in vector database
            await self.vector_db.store_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Knowledge graph processing failed: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    reasoning=f"Knowledge graph processing failed: {str(e)}"
                ),
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                metadata={"error": str(e)}
            )
    
    async def _extract_text_content(self, request: AnalysisRequest) -> str:
        """Extract text content from various data types."""
        if request.data_type == DataType.TEXT:
            return str(request.content)
        elif request.data_type == DataType.AUDIO:
            # For audio, we assume text has been extracted by audio agent
            return str(request.content)
        elif request.data_type == DataType.VIDEO:
            # For video, we assume text has been extracted by video agent
            return str(request.content)
        elif request.data_type == DataType.WEBPAGE:
            # For webpages, we assume text has been extracted by web agent
            return str(request.content)
        elif request.data_type == DataType.PDF:
            # For PDFs, we assume text has been extracted by OCR agent
            return str(request.content)
        else:
            return str(request.content)
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks using GraphRAG-inspired approach."""
        try:
            from langchain_text_splitters import TokenTextSplitter
            
            chunk_size = self.metadata.get("chunk_size", 1200)
            chunk_overlap = self.metadata.get("chunk_overlap", 100)
            
            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
            
        except ImportError:
            # Fallback to simple character-based splitting
            logger.warning("langchain_text_splitters not available, using fallback splitting")
            chunk_size = 4000  # Approximate character count for 1200 tokens
            chunk_overlap = 400  # Approximate character count for 100 tokens
            
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - chunk_overlap
                if start >= len(text):
                    break
            
            logger.info(f"Split text into {len(chunks)} chunks using fallback method")
            return chunks
    
    async def _process_text_chunks(self, text: str) -> tuple:
        """Process text in chunks and combine results (GraphRAG-inspired approach)."""
        chunks = self._split_text_into_chunks(text)
        
        all_entities = []
        all_relationships = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Extract entities and relationships from chunk using combined extraction
            extraction_result = await self.extract_entities(chunk)
            json_data = extraction_result.get("content", [{}])[0].get("json", {})
            chunk_entities = json_data.get("entities", [])
            chunk_relationships = json_data.get("relationships", [])
            
            # Add chunk metadata
            for entity in chunk_entities:
                entity["chunk_id"] = i
                entity["chunk_text"] = chunk[:100] + "..." if len(chunk) > 100 else chunk
            
            for relationship in chunk_relationships:
                relationship["chunk_id"] = i
            
            all_entities.extend(chunk_entities)
            all_relationships.extend(chunk_relationships)
        
        # Deduplicate entities based on name and type
        unique_entities = []
        seen_entities = set()
        
        for entity in all_entities:
            entity_key = (entity.get("name", ""), entity.get("type", ""))
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                unique_entities.append(entity)
        
        logger.info(f"Extracted {len(unique_entities)} unique entities from {len(chunks)} chunks")
        logger.info(f"Extracted {len(all_relationships)} relationships from {len(chunks)} chunks")
        
        return unique_entities, all_relationships
    
    async def extract_entities(self, text: str) -> dict:
        """Extract entities from text using GraphRAG-inspired approach."""
        # GraphRAG-inspired combined entity and relationship extraction prompt
        prompt = f"""
        You are an expert knowledge graph extraction system. Analyze the following text and extract both entities and their relationships with high precision.

        CRITICAL INSTRUCTIONS:
        1. First, identify ALL named entities, concepts, and important terms
        2. For each entity, you MUST categorize it into the EXACT type specified below
        3. Then, identify relationships between these entities
        4. Be extremely precise about entity types - this is crucial for knowledge graph construction

        ENTITY TYPES (choose the MOST SPECIFIC type):
        
        PERSON: 
        - Individual people, politicians, leaders, public figures
        - Examples: "Donald Trump", "Gretchen Whitmer", "Joe Biden", "President", "Governor"
        - Rule: If it's a person's name, title, or role, it's PERSON
        
        ORGANIZATION: 
        - Companies, governments, institutions, agencies, groups
        - Examples: "US Government", "White House", "Microsoft", "Media Outlets", "Administration"
        - Rule: If it's a group, institution, or organizational entity, it's ORGANIZATION
        
        LOCATION: 
        - Countries, states, cities, regions, places
        - Examples: "China", "Michigan", "New York", "American", "Chinese"
        - Rule: If it's a geographic or political location, it's LOCATION
        
        EVENT: 
        - Specific events, actions, occurrences, meetings
        - Examples: "Trade War", "Election", "Meeting", "Implementation"
        - Rule: If it's a specific happening or action, it's EVENT
        
        CONCEPT: 
        - Abstract ideas, policies, topics, theories
        - Examples: "Trade Policy", "Tariffs", "Economics", "Political Discussion"
        - Rule: If it's an abstract idea or policy, it's CONCEPT
        
        OBJECT: 
        - Physical objects, products, items
        - Examples: "iPhone", "Car", "Book", "Imports"
        - Rule: If it's a physical thing, it's OBJECT
        
        TECHNOLOGY: 
        - Tech-related terms, systems, platforms
        - Examples: "AI", "Blockchain", "Machine Learning"
        - Rule: If it's technology-related, it's TECHNOLOGY
        
        METHOD: 
        - Processes, procedures, techniques
        - Examples: "Voting System", "Analysis Method"
        - Rule: If it's a procedure or method, it's METHOD
        
        PROCESS: 
        - Ongoing activities, operations
        - Examples: "Manufacturing", "Research", "Implementation"
        - Rule: If it's an ongoing activity, it's PROCESS

        RELATIONSHIP TYPES:
        - IS_A, PART_OF, LOCATED_IN, WORKS_FOR, CREATED_BY, USES, IMPLEMENTS, SIMILAR_TO, OPPOSES, SUPPORTS, LEADS_TO, DEPENDS_ON, RELATED_TO

        Text to analyze:
        {text}

        Expected JSON format:
        {{
            "entities": [
                {{
                    "name": "entity_name",
                    "type": "entity_type",
                    "confidence": 0.95,
                    "description": "brief description"
                }}
            ],
            "relationships": [
                {{
                    "source": "entity_name",
                    "target": "entity_name", 
                    "relationship_type": "relationship_type",
                    "confidence": 0.95,
                    "description": "clear description of the relationship"
                }}
            ],
            "key_concepts": ["concept1", "concept2", "concept3"]
        }}

        IMPORTANT: 
        - Be extremely precise about entity types
        - Do NOT default to CONCEPT unless absolutely necessary
        - Consider the context and role of each entity carefully
        - Return only valid JSON, no additional text
        """
        
        try:
            response = await self.strands_agent.run(prompt)
            # Handle both string and object responses
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Try to parse as JSON with multiple fallback strategies
            json_data = None
            
            # Strategy 1: Direct JSON parsing
            try:
                json_data = json.loads(content)
            except json.JSONDecodeError:
                # Strategy 2: Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                # Strategy 3: Try to find JSON-like structure
                if not json_data:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            json_data = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            pass
            
            # Strategy 4: Enhanced fallback logic for proper entity categorization
            if not json_data or not json_data.get('entities'):
                logger.warning("Using enhanced fallback entity extraction for proper categorization")
                json_data = self._enhanced_fallback_entity_extraction(text)
            
            return {
                "content": [{
                    "json": json_data
                }]
            }
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                "content": [{
                    "json": {"entities": [], "key_concepts": []}
                }]
            }
    
    def _enhanced_fallback_entity_extraction(self, text: str) -> dict:
        """Enhanced fallback entity extraction with proper categorization using settings."""
        words = text.split()
        entities = []
        key_concepts = []
        
        # Use entity patterns from settings
        entity_types = settings.entity_categorization.entity_types
        
        # Look for capitalized words as potential entities
        for word in words:
            clean_word = word.strip('.,!?;:()[]{}"\'').strip()
            if (len(clean_word) > 2 and 
                clean_word[0].isupper() and 
                not clean_word.isupper() and  # Not all caps
                clean_word not in ['The', 'And', 'But', 'For', 'With', 'From', 'This', 'That', 'They', 'Their', 'Have', 'Been', 'Will', 'Would', 'Could', 'Should']):
                
                # Determine entity type based on settings patterns
                entity_type = "CONCEPT"  # Default
                
                for entity_category, patterns in entity_types.items():
                    if clean_word.lower() in patterns:
                        entity_type = entity_category
                        break
                
                # Additional heuristics
                if any(char.isdigit() for char in clean_word):
                    entity_type = "OBJECT"
                elif clean_word.endswith(('ing', 'tion', 'ment', 'sion', 'ance', 'ence')):
                    entity_type = "PROCESS"
                elif clean_word.endswith(('ism', 'ist', 'ity', 'ness', 'hood')):
                    entity_type = "CONCEPT"
                
                entities.append({
                    "name": clean_word,
                    "type": entity_type,
                    "confidence": 0.7,
                    "description": f"Extracted from text: {clean_word}"
                })
        
        # Add key concepts from the text
        key_concepts = [word.lower() for word in words[:15] if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'would']]
        
        return {
            "entities": entities[:15],  # Limit to 15 entities
            "key_concepts": key_concepts[:8]  # Limit to 8 concepts
        }
    
    async def map_relationships(self, text: str, entities: List[Dict]) -> dict:
        """Map relationships between entities using GraphRAG-inspired approach."""
        entity_names = [e.get("name", "") for e in entities]
        
        # GraphRAG-inspired relationship mapping prompt
        prompt = f"""
        You are an expert relationship extraction system. Analyze the relationships between entities in the given text.

        Instructions:
        1. Identify all meaningful relationships between the provided entities
        2. For each relationship, provide:
           - source: The source entity name (exact match from entities list)
           - target: The target entity name (exact match from entities list)
           - relationship_type: One of [IS_A, PART_OF, LOCATED_IN, WORKS_FOR, CREATED_BY, USES, IMPLEMENTS, SIMILAR_TO, OPPOSES, SUPPORTS, LEADS_TO, DEPENDS_ON, RELATED_TO]
           - confidence: A score between 0.0 and 1.0 based on your certainty
           - description: A clear description of the relationship
        3. Only include relationships that are explicitly mentioned or strongly implied in the text
        4. Return ONLY valid JSON in the exact format specified

        Entities to analyze: {entity_names}

        Text to analyze:
        {text}

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
        
        try:
            response = await self.strands_agent.run(prompt)
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Try to parse as JSON with multiple fallback strategies
            json_data = None
            
            # Strategy 1: Direct JSON parsing
            try:
                json_data = json.loads(content)
            except json.JSONDecodeError:
                # Strategy 2: Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                # Strategy 3: Try to find JSON-like structure
                if not json_data:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            json_data = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            pass
            
            # Strategy 4: Create structured fallback relationships
            if not json_data:
                logger.warning("JSON parsing failed, creating fallback relationships")
                relationships = []
                
                # Create simple relationships between adjacent entities
                if len(entity_names) >= 2:
                    for i in range(min(5, len(entity_names) - 1)):
                        source = entity_names[i]
                        target = entity_names[i + 1]
                        
                        # Determine relationship type based on entity types
                        source_entity = next((e for e in entities if e.get("name") == source), {})
                        target_entity = next((e for e in entities if e.get("name") == target), {})
                        
                        relationship_type = "RELATED_TO"
                        if source_entity.get("type") == "PERSON" and target_entity.get("type") == "ORGANIZATION":
                            relationship_type = "WORKS_FOR"
                        elif source_entity.get("type") == "LOCATION" and target_entity.get("type") in ["PERSON", "ORGANIZATION"]:
                            relationship_type = "LOCATED_IN"
                        elif source_entity.get("type") == "TECHNOLOGY" and target_entity.get("type") == "PROCESS":
                            relationship_type = "IMPLEMENTS"
                        
                        relationships.append({
                            "source": source,
                            "target": target,
                            "relationship_type": relationship_type,
                            "confidence": 0.4,
                            "description": f"Entities mentioned together in the text"
                        })
                
                json_data = {"relationships": relationships}
            
            return {
                "content": [{
                    "json": json_data
                }]
            }
        except Exception as e:
            logger.error(f"Relationship mapping failed: {e}")
            return {
                "content": [{
                    "json": {"relationships": []}
                }]
            }
    
    async def query_knowledge_graph(self, query: str) -> dict:
        """Query the knowledge graph for information."""
        prompt = f"""
        Query the knowledge graph for: {query}
        
        Available graph statistics:
        - Nodes: {self.graph.number_of_nodes()}
        - Edges: {self.graph.number_of_edges()}
        
        Return a JSON object with:
        - query_results: list of relevant entities and relationships
        - insights: analysis of the query results
        
        Return only valid JSON.
        """
        
        try:
            response = await self.strands_agent.run(prompt)
            # Handle both string and object responses
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Try to parse as JSON, if it fails, create mock data
            try:
                json_data = json.loads(content)
            except json.JSONDecodeError:
                # Create mock query results
                json_data = {
                    "query_results": [
                        {"entity": "Mock Entity", "type": "unknown"},
                        {"relationship": "Mock Relationship"}
                    ],
                    "insights": f"Mock analysis for query: {query}"
                }
            
            return {
                "content": [{
                    "json": json_data
                }]
            }
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return {
                "content": [{
                    "json": {"query_results": [], "insights": "Query failed"}
                }]
            }
    
    async def generate_graph_report(self, output_path: Optional[str] = None) -> dict:
        """Generate a visual graph report with both PNG and interactive HTML using settings."""
        try:
            if self.graph.number_of_nodes() == 0:
                return {
                    "content": [{
                        "json": {"message": "Graph is empty, no report generated"}
                    }]
                }
            
            # Generate timestamp for file names
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create base output path - default to Results directory using settings
            if output_path is None:
                report_filename = f"{settings.report_generation.report_filename_prefix}_{timestamp}"
                base_output_path = settings.paths.reports_dir / report_filename
            else:
                # Ensure output is in Results directory
                base_output_path = settings.paths.reports_dir / Path(output_path).name
            
            # Ensure Results directory exists
            base_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate PNG report if enabled
            png_file = None
            if settings.report_generation.generate_png:
                png_file = base_output_path.with_suffix('.png')
                await self._generate_png_report(png_file)
            
            # Generate HTML report if enabled
            html_file = None
            if settings.report_generation.generate_html:
                html_file = base_output_path.with_suffix('.html')
                await self._generate_html_report(html_file)
            
            # Generate Markdown report if enabled
            md_file = None
            if settings.report_generation.generate_md:
                md_file = base_output_path.with_suffix('.md')
                await self._generate_markdown_report(md_file)
            
            return {
                "content": [{
                    "json": {
                        "message": "Graph reports generated successfully",
                        "png_file": str(png_file) if png_file else None,
                        "html_file": str(html_file) if html_file else None,
                        "md_file": str(md_file) if md_file else None,
                        "graph_stats": self._get_graph_stats()
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Graph report generation failed: {e}")
            return {
                "content": [{
                    "json": {"error": f"Report generation failed: {str(e)}"}
                }]
            }
    
    async def _generate_png_report(self, output_file: Path):
        """Generate PNG visualization of the graph."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                             node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        
        # Add title and statistics
        plt.title(f"Knowledge Graph Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        plt.figtext(0.02, 0.02, f"Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}", 
                   fontsize=10)
        
        # Save the plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    async def _generate_html_report(self, output_file: Path):
        """Generate interactive HTML visualization of the graph."""
        # Prepare graph data for D3.js
        nodes_data = []
        edges_data = []
        
        # Process nodes
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            confidence = attrs.get('confidence', 0.5)
            
            # Enhanced group assignment based on entity type (case-insensitive)
            group = 0  # Default - Events/Objects
            node_type_lower = node_type.lower()
            
            if node_type_lower in ['person', 'people']:
                group = 0  # Red - People
            elif node_type_lower in ['organization', 'company', 'government', 'administration']:
                group = 1  # Blue - Organizations
            elif node_type_lower in ['location', 'country', 'city', 'place']:
                group = 2  # Orange - Locations
            elif node_type_lower in ['concept', 'topic', 'theme', 'method', 'technology']:
                group = 3  # Green - Concepts
            elif node_type_lower in ['event', 'action', 'object', 'process']:
                group = 4  # Purple - Events/Objects
            
            nodes_data.append({
                'id': node,
                'group': group,
                'size': max(15, int(confidence * 30)),
                'type': node_type,
                'confidence': confidence
            })
        
        # Process edges
        for source, target, attrs in self.graph.edges(data=True):
            rel_type = attrs.get('relationship_type', 'related')
            confidence = attrs.get('confidence', 0.5)
            
            edges_data.append({
                'source': source,
                'target': target,
                'value': max(1, int(confidence * 5)),
                'label': rel_type,
                'confidence': confidence
            })
        
        # Create HTML content
        html_content = self._create_html_template(nodes_data, edges_data)
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    async def _generate_markdown_report(self, output_file: Path):
        """Generate markdown report with graph analysis."""
        try:
            # Get graph statistics
            stats = self._get_graph_stats()
            
            # Generate report title
            title = f"# {settings.report_generation.report_title_prefix}\n\n"
            title += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Graph overview
            overview = "## Graph Overview\n\n"
            overview += f"- **Total Nodes:** {stats['nodes']}\n"
            overview += f"- **Total Edges:** {stats['edges']}\n"
            overview += f"- **Graph Density:** {stats['density']:.4f}\n"
            overview += f"- **Connected Components:** {stats['connected_components']}\n\n"
            
            # Entity analysis
            entity_analysis = "## Entity Analysis\n\n"
            if 'entity_types' in stats:
                entity_analysis += "### Entity Types Distribution\n\n"
                for entity_type, count in stats['entity_types'].items():
                    entity_analysis += f"- **{entity_type}:** {count} entities\n"
                entity_analysis += "\n"
            
            # Top entities
            if 'top_entities' in stats:
                entity_analysis += "### Top Entities by Connections\n\n"
                for i, (entity, connections) in enumerate(stats['top_entities'][:10], 1):
                    entity_analysis += f"{i}. **{entity}** ({connections} connections)\n"
                entity_analysis += "\n"
            
            # Relationship analysis
            relationship_analysis = "## Relationship Analysis\n\n"
            if 'relationship_types' in stats:
                relationship_analysis += "### Relationship Types\n\n"
                for rel_type, count in stats['relationship_types'].items():
                    relationship_analysis += f"- **{rel_type}:** {count} relationships\n"
                relationship_analysis += "\n"
            
            # Community analysis
            community_analysis = "## Community Analysis\n\n"
            if 'communities' in stats:
                community_analysis += f"**Number of Communities:** {len(stats['communities'])}\n\n"
                for i, community in enumerate(stats['communities'][:5], 1):
                    community_analysis += f"### Community {i}\n"
                    community_analysis += f"**Size:** {len(community)} entities\n"
                    community_analysis += f"**Entities:** {', '.join(community[:10])}"
                    if len(community) > 10:
                        community_analysis += f" (and {len(community) - 10} more)"
                    community_analysis += "\n\n"
            
            # Write markdown file
            markdown_content = title + overview + entity_analysis + relationship_analysis + community_analysis
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
                
        except Exception as e:
            logger.error(f"Failed to generate markdown report: {e}")
            # Write basic report if detailed generation fails
            basic_content = f"# {settings.report_generation.report_title_prefix}\n\n"
            basic_content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            basic_content += f"**Total Nodes:** {self.graph.number_of_nodes()}\n"
            basic_content += f"**Total Edges:** {self.graph.number_of_edges()}\n\n"
            basic_content += "*Detailed analysis could not be generated due to an error.*\n"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(basic_content)
    
    def _create_html_template(self, nodes_data, edges_data):
        """Create HTML template with D3.js visualization."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .graph-section {{
            margin-bottom: 40px;
        }}
        
        .graph-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .graph-container {{
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            padding: 20px;
            background: #f8f9fa;
            margin-bottom: 20px;
            position: relative;
        }}
        
        .zoom-indicator {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 1000;
        }}
        
        .node {{
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .node:hover {{
            stroke-width: 3px;
        }}
        
        .link {{
            stroke: #95a5a6;
            stroke-width: 2px;
            transition: all 0.3s ease;
        }}
        
        .link:hover {{
            stroke: #e74c3c;
            stroke-width: 4px;
        }}
        
        .node-label {{
            font-size: 12px;
            font-weight: bold;
            text-anchor: middle;
            pointer-events: all;
            cursor: pointer;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            z-index: 10000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            backdrop-filter: blur(5px);
        }}
        
        .modal-content {{
            background-color: white;
            margin: 5% auto;
            padding: 30px;
            border-radius: 15px;
            width: 80%;
            max-width: 600px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            position: relative;
            animation: modalSlideIn 0.3s ease-out;
        }}
        
        @keyframes modalSlideIn {{
            from {{
                opacity: 0;
                transform: translateY(-50px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #3498db;
        }}
        
        .modal-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin: 0;
        }}
        
        .close {{
            color: #aaa;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.3s;
        }}
        
        .close:hover {{
            color: #e74c3c;
        }}
        
        .node-info {{
            margin-bottom: 20px;
        }}
        
        .info-section {{
            margin-bottom: 15px;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            color: #34495e;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        
        .connections-section {{
            margin-top: 20px;
        }}
        
        .connection-item {{
            padding: 8px 12px;
            margin: 5px 0;
            background: #ecf0f1;
            border-radius: 5px;
            border-left: 4px solid #27ae60;
        }}
        
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid #333;
        }}
        
        .summary-section {{
            background: #ecf0f1;
            padding: 25px;
            border-radius: 10px;
            margin-top: 30px;
        }}
        
        .summary-title {{
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        
        .summary-list {{
            list-style: none;
            padding: 0;
        }}
        
        .summary-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #bdc3c7;
            position: relative;
            padding-left: 20px;
        }}
        
        .summary-list li:before {{
            content: "•";
            color: #3498db;
            font-weight: bold;
            position: absolute;
            left: 0;
        }}
        
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }}
        
        .btn:hover {{
            background: #2980b9;
        }}
        
        .btn.active {{
            background: #e74c3c;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Knowledge Graph Visualization</h1>
            <p>Interactive Graph Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <div class="controls">
                <button class="btn active" onclick="showGraph('main')">Main Graph</button>
                <button class="btn" onclick="showGraph('communities')">Communities</button>
                <button class="btn" onclick="showGraph('centrality')">Centrality</button>
                <button class="btn" onclick="showGraph('relationships')">Relationships</button>
                <div style="margin-top: 15px; font-size: 14px; color: #666;">
                    <strong>Zoom Controls:</strong> Mouse wheel to zoom in/out • Click and drag to pan • Double-click to reset zoom
                </div>
            </div>
            
            <div class="graph-section">
                <h2 class="graph-title" id="graph-title">Knowledge Graph Network</h2>
                <div class="graph-container">
                    <div class="zoom-indicator" id="zoom-indicator">Zoom: 100%</div>
                    <div id="graph"></div>
                </div>
                
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #e74c3c;"></div>
                        <span>People</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #3498db;"></div>
                        <span>Organizations</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f39c12;"></div>
                        <span>Locations</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #27ae60;"></div>
                        <span>Concepts</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #9b59b6;"></div>
                        <span>Events/Objects</span>
                    </div>
                </div>
            </div>
            
            <div class="summary-section">
                <h3 class="summary-title">Graph Statistics</h3>
                <ul class="summary-list">
                    <li><strong>Total Nodes:</strong> {len(nodes_data)}</li>
                    <li><strong>Total Edges:</strong> {len(edges_data)}</li>
                    <li><strong>Graph Density:</strong> {self._get_graph_stats().get('density', 0):.4f}</li>
                    <li><strong>Connected Components:</strong> {self._get_graph_stats().get('connected_components', 1)}</li>
                    <li><strong>Average Clustering:</strong> {self._get_graph_stats().get('average_clustering', 0):.4f}</li>
                    <li><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                </ul>
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <!-- Node Information Modal -->
    <div id="nodeModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title" id="modalTitle">Node Information</h2>
                <span class="close" onclick="closeModal()">&times;</span>
            </div>
            <div id="modalBody">
                <div class="node-info">
                    <div class="info-section">
                        <div class="info-label">Entity Name:</div>
                        <div class="info-value" id="nodeName">-</div>
                    </div>
                    <div class="info-section">
                        <div class="info-label">Entity Type:</div>
                        <div class="info-value" id="nodeType">-</div>
                    </div>
                    <div class="info-section">
                        <div class="info-label">Confidence Score:</div>
                        <div class="info-value" id="nodeConfidence">-</div>
                    </div>
                    <div class="info-section">
                        <div class="info-label">Group/Category:</div>
                        <div class="info-value" id="nodeGroup">-</div>
                    </div>
                    <div class="info-section">
                        <div class="info-label">Node Size:</div>
                        <div class="info-value" id="nodeSize">-</div>
                    </div>
                </div>
                <div class="connections-section">
                    <div class="info-label">Connections:</div>
                    <div id="nodeConnections">-</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Graph data
        const graphData = {{
            main: {{
                nodes: {json.dumps(nodes_data)},
                links: {json.dumps(edges_data)}
            }},
            communities: {{
                nodes: {json.dumps(nodes_data)},
                links: {json.dumps(edges_data)}
            }},
            centrality: {{
                nodes: {json.dumps(nodes_data)},
                links: {json.dumps(edges_data)}
            }},
            relationships: {{
                nodes: {json.dumps(nodes_data)},
                links: {json.dumps(edges_data)}
            }}
        }};

        const colors = ["#e74c3c", "#3498db", "#f39c12", "#27ae60", "#9b59b6"];
        let currentGraph = 'main';
        let simulation, svg, links, nodes;

        function showGraph(graphType) {{
            currentGraph = graphType;
            const data = graphData[graphType];
            
            // Update title
            const titles = {{
                main: "Knowledge Graph Network",
                communities: "Community Analysis",
                centrality: "Centrality Analysis",
                relationships: "Relationship Analysis"
            }};
            document.getElementById('graph-title').textContent = titles[graphType];
            
            // Update active button
            document.querySelectorAll('.btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // Clear existing graph
            d3.select("#graph").selectAll("*").remove();
            
            // Create new graph
            createGraph(data);
        }}

        function createGraph(data) {{
            const width = 800;
            const height = 500;
            
            // Clear existing content
            d3.select("#graph").selectAll("*").remove();
            
            // Create SVG with zoom support
            svg = d3.select("#graph")
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                .style("border", "1px solid #ddd")
                .style("background", "#f8f9fa");
            
            // Create zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10]) // Min and max zoom scale
                .on("zoom", (event) => {{
                    g.attr("transform", event.transform);
                    // Update zoom indicator
                    const zoomLevel = Math.round(event.transform.k * 100);
                    d3.select("#zoom-indicator").text(`Zoom: ${{zoomLevel}}%`);
                }});
            
            // Apply zoom to SVG
            svg.call(zoom);
            
            // Add double-click to reset zoom
            svg.on("dblclick", () => {{
                svg.transition().duration(750).call(
                    zoom.transform,
                    d3.zoomIdentity
                );
            }});
            
            // Create a group for all graph elements
            const g = svg.append("g");
            
            // Create links
            links = g.append("g")
                .selectAll("line")
                .data(data.links)
                .enter().append("line")
                .attr("class", "link")
                .style("stroke-width", d => Math.sqrt(d.value) * 2);
            
            // Create nodes
            nodes = g.append("g")
                .selectAll("circle")
                .data(data.nodes)
                .enter().append("circle")
                .attr("class", "node")
                .attr("r", d => d.size)
                .style("fill", d => colors[d.group])
                .style("stroke", "#fff")
                .style("stroke-width", 2)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            // Add labels
            const labels = g.append("g")
                .selectAll("text")
                .data(data.nodes)
                .enter().append("text")
                .attr("class", "node-label")
                .text(d => d.id)
                .style("fill", "#2c3e50");
            
            // Add link labels
            g.append("g")
                .selectAll("text")
                .data(data.links.filter(d => d.label))
                .enter().append("text")
                .text(d => d.label)
                .style("font-size", "10px")
                .style("fill", "#7f8c8d");
            
            // Create simulation
            simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .on("tick", ticked);
            
            // Add tooltip
            const tooltip = d3.select("#tooltip");
            
            nodes.on("mouseover", function(event, d) {{
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`<strong>${{d.id}}</strong><br/>Type: ${{d.type}}<br/>Confidence: ${{d.confidence.toFixed(2)}}<br/>Group: ${{d.group + 1}}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }})
            .on("mouseout", function(d) {{
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            }})
            .on("click", function(event, d) {{
                showNodeInfo(d, data);
            }});
            
            // Add same event handlers to labels
            labels.on("mouseover", function(event, d) {{
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`<strong>${{d.id}}</strong><br/>Type: ${{d.type}}<br/>Confidence: ${{d.confidence.toFixed(2)}}<br/>Group: ${{d.group + 1}}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }})
            .on("mouseout", function(d) {{
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            }})
            .on("click", function(event, d) {{
                showNodeInfo(d, data);
            }});
        }}

        function ticked() {{
            links
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            nodes
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            g.selectAll(".node-label")
                .attr("x", d => d.x)
                .attr("y", d => d.y + 5);
        }}

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        // Modal functions
        function showNodeInfo(node, graphData) {{
            // Populate modal with node information
            document.getElementById('modalTitle').textContent = `Node: ${{node.id}}`;
            document.getElementById('nodeName').textContent = node.id;
            document.getElementById('nodeType').textContent = node.type || 'Unknown';
            document.getElementById('nodeConfidence').textContent = node.confidence ? node.confidence.toFixed(3) : 'N/A';
            document.getElementById('nodeGroup').textContent = `Group ${{node.group + 1}}`;
            document.getElementById('nodeSize').textContent = node.size || 'N/A';
            
            // Find connections for this node
            const connections = graphData.links.filter(link => 
                link.source.id === node.id || link.target.id === node.id
            );
            
            let connectionsHtml = '';
            if (connections.length > 0) {{
                connections.forEach(link => {{
                    const connectedNode = link.source.id === node.id ? link.target : link.source;
                    const relationship = link.label || 'related to';
                    connectionsHtml += `<div class="connection-item">
                        <strong>${{connectedNode.id}}</strong> - ${{relationship}}
                        ${{link.confidence ? `(Confidence: ${{link.confidence.toFixed(3)}})` : ''}}
                    </div>`;
                }});
            }} else {{
                connectionsHtml = '<div class="connection-item">No direct connections found</div>';
            }}
            
            document.getElementById('nodeConnections').innerHTML = connectionsHtml;
            
            // Show modal
            document.getElementById('nodeModal').style.display = 'block';
        }}

        function closeModal() {{
            document.getElementById('nodeModal').style.display = 'none';
        }}

        // Close modal when clicking outside of it
        window.onclick = function(event) {{
            const modal = document.getElementById('nodeModal');
            if (event.target === modal) {{
                closeModal();
            }}
        }}

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});

        // Initialize with main graph
        showGraph('main');
    </script>
</body>
</html>"""
    
    async def analyze_graph_communities(self) -> dict:
        """Analyze communities in the knowledge graph using GraphRAG-inspired approach."""
        try:
            if self.graph.number_of_nodes() < 2:
                return {
                    "content": [{
                        "json": {"message": "Not enough nodes for community analysis"}
                    }]
                }
            
            # Convert to undirected graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Try multiple community detection algorithms (GraphRAG-inspired)
            community_results = {}
            
            # 1. Louvain method (modularity optimization)
            try:
                louvain_communities = nx.community.louvain_communities(undirected_graph)
                community_results["louvain"] = {
                    "method": "Louvain",
                    "communities": louvain_communities,
                    "count": len(louvain_communities)
                }
            except Exception as e:
                logger.warning(f"Louvain community detection failed: {e}")
            
            # 2. Label propagation
            try:
                label_communities = nx.community.label_propagation_communities(undirected_graph)
                community_results["label_propagation"] = {
                    "method": "Label Propagation",
                    "communities": label_communities,
                    "count": len(label_communities)
                }
            except Exception as e:
                logger.warning(f"Label propagation community detection failed: {e}")
            
            # 3. Girvan-Newman (if graph is small enough)
            try:
                if undirected_graph.number_of_nodes() <= 50:  # Only for small graphs
                    girvan_communities = nx.community.girvan_newman(undirected_graph)
                    # Get the first level of communities
                    first_level = next(girvan_communities)
                    community_results["girvan_newman"] = {
                        "method": "Girvan-Newman",
                        "communities": list(first_level),
                        "count": len(first_level)
                    }
            except Exception as e:
                logger.warning(f"Girvan-Newman community detection failed: {e}")
            
            # Use the best method (prefer Louvain)
            best_method = "louvain" if "louvain" in community_results else list(community_results.keys())[0]
            communities = community_results[best_method]["communities"]
            
            # Analyze each community with enhanced metrics
            community_analysis = []
            for i, community in enumerate(communities):
                community_nodes = list(community)
                community_subgraph = self.graph.subgraph(community_nodes)
                
                # Calculate community metrics
                density = nx.density(community_subgraph)
                avg_clustering = nx.average_clustering(community_subgraph)
                
                # Find central nodes in the community
                centrality = nx.degree_centrality(community_subgraph)
                central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                
                # Analyze entity types in the community
                entity_types = {}
                for node in community_nodes:
                    if node in self.graph.nodes:
                        node_data = self.graph.nodes[node]
                        entity_type = node_data.get("type", "unknown")
                        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                # Find most common relationships in the community
                relationship_types = {}
                for edge in community_subgraph.edges(data=True):
                    rel_type = edge[2].get("relationship_type", "unknown")
                    relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                
                analysis = {
                    "community_id": i,
                    "size": len(community_nodes),
                    "nodes": community_nodes,
                    "density": density,
                    "average_clustering": avg_clustering,
                    "central_nodes": central_nodes,
                    "entity_types": entity_types,
                    "relationship_types": relationship_types,
                    "centrality_scores": centrality
                }
                community_analysis.append(analysis)
            
            # Calculate global metrics
            modularity = nx.community.modularity(undirected_graph, communities)
            
            return {
                "content": [{
                    "json": {
                        "communities": community_analysis,
                        "total_communities": len(communities),
                        "detection_method": best_method,
                        "available_methods": list(community_results.keys()),
                        "modularity": modularity,
                        "graph_metrics": {
                            "total_nodes": self.graph.number_of_nodes(),
                            "total_edges": self.graph.number_of_edges(),
                            "average_degree": sum(dict(undirected_graph.degree()).values()) / undirected_graph.number_of_nodes() if undirected_graph.number_of_nodes() > 0 else 0
                        }
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Community analysis failed: {e}")
            return {
                "content": [{
                    "json": {"error": f"Community analysis failed: {str(e)}"}
                }]
            }
    
    async def find_entity_paths(self, source: str, target: str) -> dict:
        """Find paths between two entities in the graph."""
        try:
            if source not in self.graph or target not in self.graph:
                return {
                    "content": [{
                        "json": {"message": "One or both entities not found in graph"}
                    }]
                }
            
            # Find shortest path
            try:
                shortest_path = nx.shortest_path(self.graph, source, target)
                path_length = len(shortest_path) - 1
            except nx.NetworkXNoPath:
                shortest_path = []
                path_length = -1
            
            # Find all simple paths
            all_paths = list(nx.all_simple_paths(self.graph, source, target))
            
            return {
                "content": [{
                    "json": {
                        "source": source,
                        "target": target,
                        "shortest_path": shortest_path,
                        "shortest_path_length": path_length,
                        "all_paths_count": len(all_paths),
                        "all_paths": all_paths[:5]  # Limit to first 5 paths
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return {
                "content": [{
                    "json": {"error": f"Path finding failed: {str(e)}"}
                }]
            }
    
    async def get_entity_context(self, entity: str) -> dict:
        """Get context and connections for a specific entity."""
        try:
            if entity not in self.graph:
                return {
                    "content": [{
                        "json": {"message": "Entity not found in graph"}
                    }]
                }
            
            # Get neighbors
            neighbors = list(self.graph.neighbors(entity))
            
            # Get incoming edges
            incoming = list(self.graph.predecessors(entity))
            
            # Get outgoing edges
            outgoing = list(self.graph.successors(entity))
            
            # Get edge attributes
            edge_data = {}
            for neighbor in neighbors:
                edge_data[neighbor] = self.graph.get_edge_data(entity, neighbor)
            
            return {
                "content": [{
                    "json": {
                        "entity": entity,
                        "neighbors": neighbors,
                        "incoming_connections": incoming,
                        "outgoing_connections": outgoing,
                        "edge_data": edge_data,
                        "degree_centrality": nx.degree_centrality(self.graph).get(entity, 0)
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Entity context retrieval failed: {e}")
            return {
                "content": [{
                    "json": {"error": f"Entity context retrieval failed: {str(e)}"}
                }]
            }
    
    async def _add_to_graph(self, entities: List[Dict], relationships: List[Dict], request_id: str):
        """Add entities and relationships to the graph."""
        # Add entities as nodes
        for entity in entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "unknown")
            confidence = entity.get("confidence", 0.5)
            
            if entity_name:
                if entity_name not in self.graph:
                    self.graph.add_node(entity_name, 
                                      type=entity_type,
                                      confidence=confidence,
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
                logger.info(f"Loaded existing graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            else:
                logger.info("No existing graph found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load existing graph: {e}")
            self.graph = nx.DiGraph()
    
    def _save_graph(self):
        """Save graph to file."""
        try:
            import pickle
            with open(self.graph_file, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.debug(f"Graph saved with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
