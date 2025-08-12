"""
Duplicate Detection Service for preventing redundant processing.
Handles file-based, content-based, and similarity-based duplicate detection.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
import sqlite3
from contextlib import contextmanager

from loguru import logger

from src.config.settings import settings


@dataclass
class FileMetadata:
    """Metadata for tracking processed files."""
    file_path: str
    content_hash: str
    file_size: int
    modification_time: float
    first_processed: datetime
    last_processed: datetime
    processing_count: int
    data_type: str
    agent_id: str
    result_id: str


@dataclass
class DuplicateDetectionResult:
    """Result of duplicate detection."""
    is_duplicate: bool
    duplicate_type: Optional[str]  # 'exact', 'similar', 'file_path', 'content_hash'
    confidence: float
    existing_metadata: Optional[FileMetadata]
    similarity_score: Optional[float]
    recommendation: str  # 'skip', 'update', 'reprocess'


class DuplicateDetectionService:
    """Service for detecting and managing duplicates in the sentiment analysis system."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(
            db_path or settings.paths.cache_dir / "duplicate_detection.db"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Configuration
        self.similarity_threshold = 0.95  # 95% similarity for near-duplicates
        self.max_file_size = 100 * 1024 * 1024  # 100MB max file size
        
        logger.info(f"Duplicate Detection Service initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database for tracking processed files."""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    modification_time REAL NOT NULL,
                    first_processed TEXT NOT NULL,
                    last_processed TEXT NOT NULL,
                    processing_count INTEGER DEFAULT 1,
                    data_type TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    result_id TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON processed_files(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON processed_files(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_data_type ON processed_files(data_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON processed_files(agent_id)")
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get a database connection with proper error handling."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file content."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                logger.warning(f"File {file_path} is too large ({file_size} bytes), using path-based hash")
                return self._compute_path_hash(str(file_path))
            
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logger.error(f"Error computing file hash for {file_path}: {e}")
            return self._compute_path_hash(str(file_path))
    
    def _compute_path_hash(self, file_path: str) -> str:
        """Compute hash based on file path and modification time."""
        stat = os.stat(file_path)
        content = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _compute_content_hash(self, content: str) -> str:
        """Compute hash of text content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def detect_duplicates(
        self,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        data_type: str = "unknown",
        agent_id: str = "unknown",
        force_reprocess: bool = False
    ) -> DuplicateDetectionResult:
        """
        Detect duplicates based on file path, content hash, or similarity.
        
        Args:
            file_path: Path to the file being processed
            content: Text content being processed
            data_type: Type of data being processed
            agent_id: ID of the processing agent
            force_reprocess: Whether to force reprocessing even if duplicate found
            
        Returns:
            DuplicateDetectionResult with detection information
        """
        try:
            # If force reprocess is enabled, skip duplicate detection
            if force_reprocess:
                return DuplicateDetectionResult(
                    is_duplicate=False,
                    duplicate_type=None,
                    confidence=0.0,
                    existing_metadata=None,
                    similarity_score=None,
                    recommendation="reprocess"
                )
            
            # Check file path duplicates
            if file_path:
                file_duplicate = await self._check_file_path_duplicate(file_path, data_type)
                if file_duplicate.is_duplicate:
                    return file_duplicate
            
            # Check content hash duplicates
            if content:
                content_duplicate = await self._check_content_hash_duplicate(content, data_type)
                if content_duplicate.is_duplicate:
                    return content_duplicate
            
            # Check for similar content (if we have content to compare)
            if content:
                similar_duplicate = await self._check_similar_content(content, data_type)
                if similar_duplicate.is_duplicate:
                    return similar_duplicate
            
            # No duplicates found
            return DuplicateDetectionResult(
                is_duplicate=False,
                duplicate_type=None,
                confidence=0.0,
                existing_metadata=None,
                similarity_score=None,
                recommendation="process"
            )
            
        except Exception as e:
            logger.error(f"Error in duplicate detection: {e}")
            # On error, allow processing to continue
            return DuplicateDetectionResult(
                is_duplicate=False,
                duplicate_type=None,
                confidence=0.0,
                existing_metadata=None,
                similarity_score=None,
                recommendation="process"
            )
    
    async def _check_file_path_duplicate(self, file_path: str, data_type: str) -> DuplicateDetectionResult:
        """Check if file path has been processed before."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM processed_files 
                    WHERE file_path = ? AND data_type = ?
                    ORDER BY last_processed DESC
                    LIMIT 1
                """, (file_path, data_type))
                
                row = cursor.fetchone()
                if row:
                    # Check if file has been modified since last processing
                    current_mtime = os.path.getmtime(file_path)
                    if current_mtime <= row['modification_time']:
                        # File hasn't changed, exact duplicate
                        metadata = FileMetadata(
                            file_path=row['file_path'],
                            content_hash=row['content_hash'],
                            file_size=row['file_size'],
                            modification_time=row['modification_time'],
                            first_processed=datetime.fromisoformat(row['first_processed']),
                            last_processed=datetime.fromisoformat(row['last_processed']),
                            processing_count=row['processing_count'],
                            data_type=row['data_type'],
                            agent_id=row['agent_id'],
                            result_id=row['result_id']
                        )
                        
                        return DuplicateDetectionResult(
                            is_duplicate=True,
                            duplicate_type="file_path",
                            confidence=1.0,
                            existing_metadata=metadata,
                            similarity_score=1.0,
                            recommendation="skip"
                        )
                    else:
                        # File has been modified, recommend update
                        return DuplicateDetectionResult(
                            is_duplicate=True,
                            duplicate_type="file_path_modified",
                            confidence=0.8,
                            existing_metadata=None,
                            similarity_score=0.8,
                            recommendation="update"
                        )
            
            return DuplicateDetectionResult(
                is_duplicate=False,
                duplicate_type=None,
                confidence=0.0,
                existing_metadata=None,
                similarity_score=None,
                recommendation="process"
            )
            
        except Exception as e:
            logger.error(f"Error checking file path duplicate: {e}")
            return DuplicateDetectionResult(
                is_duplicate=False,
                duplicate_type=None,
                confidence=0.0,
                existing_metadata=None,
                similarity_score=None,
                recommendation="process"
            )
    
    async def _check_content_hash_duplicate(self, content: str, data_type: str) -> DuplicateDetectionResult:
        """Check if content hash has been processed before."""
        try:
            content_hash = self._compute_content_hash(content)
            
            with self._get_db_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM processed_files 
                    WHERE content_hash = ? AND data_type = ?
                    ORDER BY last_processed DESC
                    LIMIT 1
                """, (content_hash, data_type))
                
                row = cursor.fetchone()
                if row:
                    metadata = FileMetadata(
                        file_path=row['file_path'],
                        content_hash=row['content_hash'],
                        file_size=row['file_size'],
                        modification_time=row['modification_time'],
                        first_processed=datetime.fromisoformat(row['first_processed']),
                        last_processed=datetime.fromisoformat(row['last_processed']),
                        processing_count=row['processing_count'],
                        data_type=row['data_type'],
                        agent_id=row['agent_id'],
                        result_id=row['result_id']
                    )
                    
                    return DuplicateDetectionResult(
                        is_duplicate=True,
                        duplicate_type="content_hash",
                        confidence=1.0,
                        existing_metadata=metadata,
                        similarity_score=1.0,
                        recommendation="skip"
                    )
            
            return DuplicateDetectionResult(
                is_duplicate=False,
                duplicate_type=None,
                confidence=0.0,
                existing_metadata=None,
                similarity_score=None,
                recommendation="process"
            )
            
        except Exception as e:
            logger.error(f"Error checking content hash duplicate: {e}")
            return DuplicateDetectionResult(
                is_duplicate=False,
                duplicate_type=None,
                confidence=0.0,
                existing_metadata=None,
                similarity_score=None,
                recommendation="process"
            )
    
    async def _check_similar_content(self, content: str, data_type: str) -> DuplicateDetectionResult:
        """Check for similar content using vector similarity (placeholder for now)."""
        # This is a placeholder - in a full implementation, you would:
        # 1. Use the vector database to find similar embeddings
        # 2. Compare content similarity scores
        # 3. Return results above the similarity threshold
        
        # For now, return no similarity found
        return DuplicateDetectionResult(
            is_duplicate=False,
            duplicate_type=None,
            confidence=0.0,
            existing_metadata=None,
            similarity_score=None,
            recommendation="process"
        )
    
    async def record_processing(
        self,
        file_path: Optional[str],
        content: Optional[str],
        data_type: str,
        agent_id: str,
        result_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record that a file/content has been processed."""
        try:
            # Compute content hash
            if content:
                content_hash = self._compute_content_hash(content)
            elif file_path:
                content_hash = self._compute_file_hash(file_path)
            else:
                content_hash = "unknown"
            
            # Get file information
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                modification_time = os.path.getmtime(file_path)
            else:
                file_size = len(content) if content else 0
                modification_time = datetime.now().timestamp()
            
            current_time = datetime.now().isoformat()
            
            with self._get_db_connection() as conn:
                # Check if record already exists
                cursor = conn.execute("""
                    SELECT id, processing_count FROM processed_files 
                    WHERE file_path = ? AND data_type = ?
                """, (file_path or "content_only", data_type))
                
                row = cursor.fetchone()
                if row:
                    # Update existing record
                    conn.execute("""
                        UPDATE processed_files 
                        SET content_hash = ?, file_size = ?, modification_time = ?,
                            last_processed = ?, processing_count = ?, agent_id = ?, 
                            result_id = ?, metadata = ?
                        WHERE id = ?
                    """, (
                        content_hash, file_size, modification_time, current_time,
                        row['processing_count'] + 1, agent_id, result_id,
                        json.dumps(metadata) if metadata else None, row['id']
                    ))
                else:
                    # Insert new record
                    conn.execute("""
                        INSERT INTO processed_files 
                        (file_path, content_hash, file_size, modification_time,
                         first_processed, last_processed, processing_count,
                         data_type, agent_id, result_id, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file_path or "content_only", content_hash, file_size, modification_time,
                        current_time, current_time, 1, data_type, agent_id, result_id,
                        json.dumps(metadata) if metadata else None
                    ))
                
                conn.commit()
                logger.info(f"Recorded processing for {file_path or 'content'}")
                
        except Exception as e:
            logger.error(f"Error recording processing: {e}")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed files."""
        try:
            with self._get_db_connection() as conn:
                # Total processed files
                cursor = conn.execute("SELECT COUNT(*) as total FROM processed_files")
                total_files = cursor.fetchone()['total']
                
                # Files by data type
                cursor = conn.execute("""
                    SELECT data_type, COUNT(*) as count 
                    FROM processed_files 
                    GROUP BY data_type
                """)
                files_by_type = {row['data_type']: row['count'] for row in cursor.fetchall()}
                
                # Most processed files
                cursor = conn.execute("""
                    SELECT file_path, processing_count, last_processed
                    FROM processed_files 
                    ORDER BY processing_count DESC 
                    LIMIT 10
                """)
                most_processed = [
                    {
                        'file_path': row['file_path'],
                        'processing_count': row['processing_count'],
                        'last_processed': row['last_processed']
                    }
                    for row in cursor.fetchall()
                ]
                
                # Recent processing activity
                cursor = conn.execute("""
                    SELECT COUNT(*) as recent_count 
                    FROM processed_files 
                    WHERE last_processed > datetime('now', '-24 hours')
                """)
                recent_activity = cursor.fetchone()['recent_count']
                
                return {
                    'total_files': total_files,
                    'files_by_type': files_by_type,
                    'most_processed': most_processed,
                    'recent_activity_24h': recent_activity
                }
                
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}
    
    async def cleanup_old_records(self, days_old: int = 30):
        """Clean up old processing records."""
        try:
            with self._get_db_connection() as conn:
                conn.execute("""
                    DELETE FROM processed_files 
                    WHERE last_processed < datetime('now', '-{} days')
                """.format(days_old))
                
                deleted_count = conn.total_changes
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old processing records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return 0
