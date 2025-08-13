"""
Database Connector

Provides comprehensive database integration capabilities including:
- Multiple database type support (PostgreSQL, MySQL, SQLite, MongoDB)
- Connection pooling and management
- Query optimization and caching
- Data validation and transformation
- Error handling and logging
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import sqlite3
from contextlib import asynccontextmanager
import aiomysql
import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"


@dataclass
class DatabaseConfig:
    """Configuration for a database connection"""
    name: str
    db_type: DatabaseType
    host: str = "localhost"
    port: int = None
    database: str = ""
    username: str = ""
    password: str = ""
    connection_string: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    timeout: int = 30
    ssl_mode: str = "prefer"
    charset: str = "utf8mb4"


class DatabaseConnector:
    """
    Manages multiple database connections with comprehensive features
    """
    
    def __init__(self, configs: Optional[Dict[str, DatabaseConfig]] = None):
        """
        Initialize the database connector
        
        Args:
            configs: Dictionary of database configurations
        """
        self.configs = configs or {}
        self.connections = {}
        self.pools = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the database connector"""
        logging.basicConfig(level=logging.INFO)
        
    def add_database_config(self, name: str, config: DatabaseConfig):
        """
        Add a new database configuration
        
        Args:
            name: Name identifier for the database
            config: Database configuration object
        """
        self.configs[name] = config
        logger.info(f"Added database configuration for: {name}")
        
    def remove_database_config(self, name: str):
        """
        Remove a database configuration
        
        Args:
            name: Name of the database configuration to remove
        """
        if name in self.configs:
            del self.configs[name]
            if name in self.connections:
                del self.connections[name]
            if name in self.pools:
                del self.pools[name]
            logger.info(f"Removed database configuration for: {name}")
            
    def _get_connection_string(self, config: DatabaseConfig) -> str:
        """
        Generate connection string based on database type
        
        Args:
            config: Database configuration
            
        Returns:
            Connection string
        """
        if config.connection_string:
            return config.connection_string
            
        if config.db_type == DatabaseType.SQLITE:
            return f"sqlite:///{config.database}"
        elif config.db_type == DatabaseType.POSTGRESQL:
            port = config.port or 5432
            return (f"postgresql+asyncpg://{config.username}:{config.password}"
                   f"@{config.host}:{port}/{config.database}")
        elif config.db_type == DatabaseType.MYSQL:
            port = config.port or 3306
            return (f"mysql+aiomysql://{config.username}:{config.password}"
                   f"@{config.host}:{port}/{config.database}")
        else:
            raise ValueError(f"Unsupported database type: {config.db_type}")
            
    async def _create_connection_pool(self, name: str, config: DatabaseConfig):
        """
        Create a connection pool for the database
        
        Args:
            name: Database name
            config: Database configuration
        """
        try:
            if config.db_type == DatabaseType.SQLITE:
                # SQLite doesn't need connection pooling
                self.connections[name] = sqlite3.connect(config.database)
                
            elif config.db_type == DatabaseType.POSTGRESQL:
                self.pools[name] = await asyncpg.create_pool(
                    host=config.host,
                    port=config.port or 5432,
                    user=config.username,
                    password=config.password,
                    database=config.database,
                    min_size=5,
                    max_size=config.pool_size,
                    command_timeout=config.timeout
                )
                
            elif config.db_type == DatabaseType.MYSQL:
                self.pools[name] = await aiomysql.create_pool(
                    host=config.host,
                    port=config.port or 3306,
                    user=config.username,
                    password=config.password,
                    db=config.database,
                    charset=config.charset,
                    autocommit=True,
                    minsize=5,
                    maxsize=config.pool_size
                )
                
            elif config.db_type == DatabaseType.MONGODB:
                client = AsyncIOMotorClient(
                    config.connection_string or f"mongodb://{config.host}:{config.port or 27017}"
                )
                self.connections[name] = client[config.database]
                
            logger.info(f"Created connection pool for database: {name}")
            
        except Exception as e:
            logger.error(f"Failed to create connection pool for {name}: {str(e)}")
            raise
            
    async def get_connection(self, name: str):
        """
        Get a database connection
        
        Args:
            name: Database name
            
        Returns:
            Database connection object
        """
        if name not in self.configs:
            raise ValueError(f"Database configuration '{name}' not found")
            
        config = self.configs[name]
        
        # Create pool if it doesn't exist
        if name not in self.pools and name not in self.connections:
            await self._create_connection_pool(name, config)
            
        if config.db_type == DatabaseType.SQLITE:
            return self.connections[name]
        elif config.db_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
            return await self.pools[name].acquire()
        elif config.db_type == DatabaseType.MONGODB:
            return self.connections[name]
            
    async def execute_query(self, db_name: str, query: str, 
                           params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a database query
        
        Args:
            db_name: Database name
            query: SQL query or MongoDB query
            params: Query parameters
            
        Returns:
            Query results
        """
        if db_name not in self.configs:
            raise ValueError(f"Database configuration '{db_name}' not found")
            
        config = self.configs[db_name]
        conn = await self.get_connection(db_name)
        
        try:
            if config.db_type == DatabaseType.SQLITE:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                    
                if query.strip().upper().startswith('SELECT'):
                    columns = [description[0] for description in cursor.description]
                    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                else:
                    conn.commit()
                    results = [{'affected_rows': cursor.rowcount}]
                    
            elif config.db_type == DatabaseType.POSTGRESQL:
                async with conn.transaction():
                    if params:
                        results = await conn.fetch(query, *params.values())
                    else:
                        results = await conn.fetch(query)
                        
                if isinstance(results, list):
                    results = [dict(row) for row in results]
                    
            elif config.db_type == DatabaseType.MYSQL:
                async with conn.cursor() as cursor:
                    if params:
                        await cursor.execute(query, params)
                    else:
                        await cursor.execute(query)
                        
                    if query.strip().upper().startswith('SELECT'):
                        columns = [desc[0] for desc in cursor.description]
                        rows = await cursor.fetchall()
                        results = [dict(zip(columns, row)) for row in rows]
                    else:
                        results = [{'affected_rows': cursor.rowcount}]
                        
            elif config.db_type == DatabaseType.MONGODB:
                # For MongoDB, query should be a dict with collection and operation
                if isinstance(query, dict):
                    collection_name = query.get('collection')
                    operation = query.get('operation')
                    filter_query = query.get('filter', {})
                    projection = query.get('projection')
                    
                    collection = conn[collection_name]
                    
                    if operation == 'find':
                        cursor = collection.find(filter_query, projection)
                        results = await cursor.to_list(length=None)
                    elif operation == 'insert_one':
                        result = await collection.insert_one(filter_query)
                        results = [{'inserted_id': str(result.inserted_id)}]
                    elif operation == 'update_one':
                        result = await collection.update_one(
                            filter_query, query.get('update', {})
                        )
                        results = [{'modified_count': result.modified_count}]
                    elif operation == 'delete_one':
                        result = await collection.delete_one(filter_query)
                        results = [{'deleted_count': result.deleted_count}]
                    else:
                        raise ValueError(f"Unsupported MongoDB operation: {operation}")
                else:
                    raise ValueError("MongoDB queries must be dictionaries")
                    
            logger.info(f"Query executed successfully on database: {db_name}")
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed on {db_name}: {str(e)}")
            raise
        finally:
            if config.db_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
                self.pools[db_name].release(conn)
                
    async def execute_batch(self, db_name: str, queries: List[Dict]) -> List[Dict]:
        """
        Execute multiple queries in batch
        
        Args:
            db_name: Database name
            queries: List of query dictionaries
            
        Returns:
            List of results
        """
        results = []
        for query_data in queries:
            try:
                query = query_data.get('query')
                params = query_data.get('params')
                result = await self.execute_query(db_name, query, params)
                results.append({
                    'success': True,
                    'result': result,
                    'query': query_data
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'query': query_data
                })
                
        return results
        
    async def get_table_schema(self, db_name: str, table_name: str) -> Dict:
        """
        Get table schema information
        
        Args:
            db_name: Database name
            table_name: Table name
            
        Returns:
            Schema information
        """
        config = self.configs[db_name]
        
        if config.db_type == DatabaseType.SQLITE:
            query = "PRAGMA table_info(?)"
            columns = await self.execute_query(db_name, query, [table_name])
            return {
                'table_name': table_name,
                'columns': columns
            }
        elif config.db_type == DatabaseType.POSTGRESQL:
            query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position
            """
            columns = await self.execute_query(db_name, query, [table_name])
            return {
                'table_name': table_name,
                'columns': columns
            }
        elif config.db_type == DatabaseType.MYSQL:
            query = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION
            """
            columns = await self.execute_query(db_name, query, [table_name])
            return {
                'table_name': table_name,
                'columns': columns
            }
        elif config.db_type == DatabaseType.MONGODB:
            # MongoDB doesn't have a fixed schema, return collection info
            conn = await self.get_connection(db_name)
            collection = conn[table_name]
            count = await collection.count_documents({})
            return {
                'collection_name': table_name,
                'document_count': count,
                'schema_flexible': True
            }
            
    def get_database_status(self, db_name: str) -> Dict[str, Any]:
        """
        Get status information for a database
        
        Args:
            db_name: Database name
            
        Returns:
            Status information dictionary
        """
        if db_name not in self.configs:
            return {'error': f"Database '{db_name}' not found"}
            
        config = self.configs[db_name]
        
        return {
            'name': db_name,
            'type': config.db_type.value,
            'host': config.host,
            'database': config.database,
            'pool_size': config.pool_size,
            'has_pool': db_name in self.pools,
            'has_connection': db_name in self.connections
        }
        
    async def close_all(self):
        """Close all database connections and pools"""
        for name, pool in self.pools.items():
            await pool.close()
            
        for name, conn in self.connections.items():
            if hasattr(conn, 'close'):
                conn.close()
                
        logger.info("All database connections closed")
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()
