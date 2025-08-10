"""
Project settings and configuration for the sentiment analysis system.
"""

from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class EntityCategorizationConfig(BaseModel):
    """Configuration for entity categorization in knowledge graphs."""
    
    # Entity types with improved categorization
    entity_types: Dict[str, List[str]] = Field(default_factory=lambda: {
        "PERSON": [
            "trump", "donald", "biden", "joe", "obama", "clinton", "whitmer", "gretchen",
            "president", "governor", "senator", "congressman", "congresswoman", "mayor",
            "john", "mike", "david", "sarah", "emma", "alex", "chris", "james", "michael",
            "william", "robert", "thomas", "daniel", "matthew", "anthony", "mark", "donald",
            "steven", "paul", "andrew", "joshua", "kenneth", "kevin", "brian", "george",
            "timothy", "ronald", "jason", "edward", "jeffrey", "ryan", "jacob", "gary",
            "nicholas", "eric", "jonathan", "stephen", "larry", "justin", "scott", "brandon",
            "benjamin", "frank", "gregory", "raymond", "samuel", "patrick", "alexander",
            "jack", "dennis", "jerry", "tyler", "aaron", "jose", "adam", "nathan", "henry",
            "douglas", "zachary", "peter", "kyle", "walter", "ethan", "jeremy", "harold",
            "succeeded", "elected", "appointed", "nominated", "confirmed", "inaugurated"
        ],
        "LOCATION": [
            "michigan", "california", "texas", "florida", "new york", "washington", "usa",
            "china", "chinese", "mexico", "american", "canada", "britain", "france", "germany",
            "japan", "india", "russia", "brazil", "australia", "italy", "spain", "korea",
            "thailand", "vietnam", "singapore", "malaysia", "indonesia", "philippines",
            "europe", "asia", "africa", "america", "north", "south", "east", "west",
            "atlantic", "pacific", "mediterranean", "caribbean", "gulf", "bay", "river",
            "mountain", "valley", "desert", "forest", "ocean", "sea", "lake", "island",
            "peninsula", "isthmus", "strait", "canal", "bridge", "tunnel", "highway",
            "street", "avenue", "boulevard", "road", "drive", "lane", "place", "court"
        ],
        "ORGANIZATION": [
            "government", "administration", "congress", "senate", "house", "white house",
            "federal", "state", "local", "municipal", "county", "city", "town", "village",
            "company", "corporation", "corp", "inc", "llc", "ltd", "co", "associates",
            "enterprises", "industries", "manufacturing", "production", "services",
            "media", "outlets", "news", "press", "journal", "magazine", "newspaper",
            "television", "radio", "broadcast", "network", "channel", "station",
            "university", "college", "school", "institute", "academy", "foundation",
            "association", "society", "club", "organization", "agency", "department",
            "ministry", "bureau", "office", "commission", "committee", "board", "council"
        ],
        "CONCEPT": [
            "tariffs", "policy", "trade", "economics", "discussion", "articles", "language",
            "politics", "democracy", "republic", "freedom", "liberty", "justice", "equality",
            "rights", "law", "legislation", "regulation", "tax", "taxation", "budget",
            "finance", "banking", "investment", "market", "economy", "business", "commerce",
            "industry", "manufacturing", "agriculture", "technology", "innovation",
            "research", "development", "education", "healthcare", "medicine", "science",
            "environment", "climate", "energy", "transportation", "infrastructure",
            "security", "defense", "military", "diplomacy", "foreign", "international",
            "global", "national", "regional", "local", "urban", "rural", "suburban"
        ],
        "OBJECT": [
            "imports", "exports", "products", "goods", "materials", "resources",
            "equipment", "machinery", "vehicles", "automobiles", "cars", "trucks",
            "electronics", "computers", "phones", "devices", "appliances", "furniture",
            "clothing", "textiles", "food", "agricultural", "chemicals", "pharmaceuticals",
            "weapons", "ammunition", "tools", "instruments", "machines", "engines",
            "batteries", "fuels", "energy", "electricity", "gas", "oil", "coal",
            "steel", "aluminum", "copper", "iron", "gold", "silver", "platinum"
        ],
        "PROCESS": [
            "implementation", "criticism", "analysis", "evaluation", "assessment",
            "planning", "development", "production", "manufacturing", "distribution",
            "marketing", "advertising", "promotion", "sales", "purchase", "acquisition",
            "merger", "acquisition", "expansion", "growth", "development", "improvement",
            "maintenance", "repair", "upgrade", "modernization", "innovation", "research",
            "testing", "quality", "control", "monitoring", "supervision", "management",
            "administration", "coordination", "collaboration", "cooperation", "partnership",
            "negotiation", "agreement", "contract", "treaty", "alliance", "coalition"
        ]
    })
    
    # Relationship types
    relationship_types: List[str] = Field(default=[
        "IS_A", "PART_OF", "LOCATED_IN", "WORKS_FOR", "CREATED_BY", "USES", 
        "IMPLEMENTS", "SIMILAR_TO", "OPPOSES", "SUPPORTS", "LEADS_TO", 
        "DEPENDS_ON", "RELATED_TO"
    ])
    
    # Confidence thresholds
    min_confidence: float = 0.7
    default_confidence: float = 0.8


class ReportGenerationConfig(BaseModel):
    """Configuration for report generation."""
    
    # Output directories
    results_dir: Path = Field(default=Path("./Results"))
    docs_dir: Path = Field(default=Path("./docs"))
    test_dir: Path = Field(default=Path("./Test"))
    
    # Report file naming
    report_title_prefix: str = "Knowledge Graph Analysis Report"
    report_filename_prefix: str = "knowledge_graph_report"
    
    # Report formats
    generate_html: bool = True
    generate_md: bool = True
    generate_png: bool = True
    
    # Report content
    include_graph_stats: bool = True
    include_entity_analysis: bool = True
    include_relationship_analysis: bool = True
    include_community_analysis: bool = True
    
    # Graph visualization
    graph_layout: str = "spring"  # spring, circular, random, shell
    node_size: int = 300
    edge_width: float = 1.0
    figure_size: tuple = (12, 8)
    dpi: int = 300


class ProjectPathsConfig(BaseModel):
    """Configuration for project paths and directories."""
    
    # Base directories
    base_dir: Path = Field(default=Path("."))
    src_dir: Path = Field(default=Path("./src"))
    results_dir: Path = Field(default=Path("./Results"))
    docs_dir: Path = Field(default=Path("./docs"))
    test_dir: Path = Field(default=Path("./Test"))
    examples_dir: Path = Field(default=Path("./examples"))
    scripts_dir: Path = Field(default=Path("./scripts"))
    
    # Subdirectories
    knowledge_graphs_dir: Path = Field(default=Path("./Results/knowledge_graphs"))
    reports_dir: Path = Field(default=Path("./Results/reports"))
    temp_dir: Path = Field(default=Path("./temp"))
    cache_dir: Path = Field(default=Path("./cache"))
    models_dir: Path = Field(default=Path("./models"))
    
    # File patterns
    report_file_patterns: List[str] = Field(default=[
        "*.html", "*.md", "*.png", "*.jpg", "*.jpeg"
    ])
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.results_dir,
            self.docs_dir,
            self.test_dir,
            self.knowledge_graphs_dir,
            self.reports_dir,
            self.temp_dir,
            self.cache_dir,
            self.models_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """Main settings configuration."""
    
    # Entity categorization
    entity_categorization: EntityCategorizationConfig = Field(
        default_factory=EntityCategorizationConfig
    )
    
    # Report generation
    report_generation: ReportGenerationConfig = Field(
        default_factory=ReportGenerationConfig
    )
    
    # Project paths
    paths: ProjectPathsConfig = Field(
        default_factory=ProjectPathsConfig
    )
    
    # Python executable path
    python_executable: str = "./venv/Scripts/python.exe"
    
    # Test settings
    test_timeout: int = 300  # 5 minutes
    test_retries: int = 3
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.paths.ensure_directories()


# Global settings instance
settings = Settings()
