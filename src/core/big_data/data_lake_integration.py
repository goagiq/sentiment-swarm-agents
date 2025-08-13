"""
Data Lake Integration

Data lake integration for cloud storage platforms:
- AWS S3 integration
- Azure Blob Storage
- Google Cloud Storage
- Data lake management
"""

from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import os
from pathlib import Path

from loguru import logger

from src.config.big_data_config import get_big_data_config
from src.core.error_handling_service import ErrorHandlingService


@dataclass
class DataLakeConfig:
    """Configuration for data lake connection."""
    provider: str  # 's3', 'azure', 'gcs'
    bucket_name: str
    region: str
    credentials: Dict[str, str] = field(default_factory=dict)
    endpoint_url: Optional[str] = None


@dataclass
class DataLakeObject:
    """Represents a data lake object."""
    key: str
    size: int
    last_modified: datetime
    metadata: Dict[str, str] = field(default_factory=dict)
    etag: Optional[str] = None


class DataLakeIntegration:
    """
    Data lake integration for cloud storage platforms.
    """
    
    def __init__(self):
        """Initialize data lake integration."""
        self.config = get_big_data_config()
        self.error_handler = ErrorHandlingService()
        
        # Client connections
        self.s3_client = None
        self.azure_client = None
        self.gcs_client = None
        
        # Active connections
        self.active_connections: Dict[str, Any] = {}
        
        logger.info("DataLakeIntegration initialized")
    
    async def connect_s3(self, config: DataLakeConfig) -> bool:
        """Connect to AWS S3."""
        try:
            # Conditional import for boto3
            try:
                import boto3
                from botocore.exceptions import ClientError
            except ImportError:
                logger.warning("boto3 not available. Install with: pip install boto3")
                return False
            
            # Create S3 client
            self.s3_client = boto3.client(
                's3',
                region_name=config.region,
                aws_access_key_id=config.credentials.get('access_key_id'),
                aws_secret_access_key=config.credentials.get('secret_access_key'),
                endpoint_url=config.endpoint_url
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=config.bucket_name)
            
            self.active_connections['s3'] = {
                'client': self.s3_client,
                'config': config
            }
            
            logger.info(f"Connected to S3 bucket: {config.bucket_name}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "s3_connection_error",
                f"Failed to connect to S3: {str(e)}",
                error_data={'bucket': config.bucket_name, 'error': str(e)}
            )
            return False
    
    async def connect_azure(self, config: DataLakeConfig) -> bool:
        """Connect to Azure Blob Storage."""
        try:
            # Conditional import for azure-storage-blob
            try:
                from azure.storage.blob import BlobServiceClient
            except ImportError:
                logger.warning(
                    "azure-storage-blob not available. "
                    "Install with: pip install azure-storage-blob"
                )
                return False
            
            # Create connection string
            connection_string = config.credentials.get('connection_string')
            if not connection_string:
                logger.error("Azure connection string not provided")
                return False
            
            # Create blob service client
            self.azure_client = BlobServiceClient.from_connection_string(
                connection_string
            )
            
            # Test connection
            container_client = self.azure_client.get_container_client(
                config.bucket_name
            )
            container_client.get_container_properties()
            
            self.active_connections['azure'] = {
                'client': self.azure_client,
                'config': config
            }
            
            logger.info(f"Connected to Azure container: {config.bucket_name}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "azure_connection_error",
                f"Failed to connect to Azure: {str(e)}",
                error_data={'container': config.bucket_name, 'error': str(e)}
            )
            return False
    
    async def connect_gcs(self, config: DataLakeConfig) -> bool:
        """Connect to Google Cloud Storage."""
        try:
            # Conditional import for google-cloud-storage
            try:
                from google.cloud import storage
            except ImportError:
                logger.warning(
                    "google-cloud-storage not available. "
                    "Install with: pip install google-cloud-storage"
                )
                return False
            
            # Create storage client
            self.gcs_client = storage.Client()
            
            # Test connection
            bucket = self.gcs_client.bucket(config.bucket_name)
            bucket.reload()
            
            self.active_connections['gcs'] = {
                'client': self.gcs_client,
                'config': config
            }
            
            logger.info(f"Connected to GCS bucket: {config.bucket_name}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "gcs_connection_error",
                f"Failed to connect to GCS: {str(e)}",
                error_data={'bucket': config.bucket_name, 'error': str(e)}
            )
            return False
    
    async def upload_file(self, provider: str, local_path: str, 
                         remote_key: str) -> bool:
        """Upload a file to data lake."""
        try:
            if provider == 's3' and 's3' in self.active_connections:
                return await self._upload_to_s3(local_path, remote_key)
            elif provider == 'azure' and 'azure' in self.active_connections:
                return await self._upload_to_azure(local_path, remote_key)
            elif provider == 'gcs' and 'gcs' in self.active_connections:
                return await self._upload_to_gcs(local_path, remote_key)
            else:
                logger.error(f"Provider {provider} not connected")
                return False
                
        except Exception as e:
            await self.error_handler.handle_error(
                "upload_error",
                f"Failed to upload file: {str(e)}",
                error_data={'provider': provider, 'file': local_path, 'error': str(e)}
            )
            return False
    
    async def _upload_to_s3(self, local_path: str, remote_key: str) -> bool:
        """Upload file to S3."""
        try:
            connection = self.active_connections['s3']
            client = connection['client']
            bucket = connection['config'].bucket_name
            
            client.upload_file(local_path, bucket, remote_key)
            logger.info(f"Uploaded {local_path} to S3: {remote_key}")
            return True
            
        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
            return False
    
    async def _upload_to_azure(self, local_path: str, remote_key: str) -> bool:
        """Upload file to Azure Blob Storage."""
        try:
            connection = self.active_connections['azure']
            client = connection['client']
            container = connection['config'].bucket_name
            
            blob_client = client.get_blob_client(
                container=container, blob=remote_key
            )
            
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"Uploaded {local_path} to Azure: {remote_key}")
            return True
            
        except Exception as e:
            logger.error(f"Azure upload failed: {str(e)}")
            return False
    
    async def _upload_to_gcs(self, local_path: str, remote_key: str) -> bool:
        """Upload file to Google Cloud Storage."""
        try:
            connection = self.active_connections['gcs']
            client = connection['client']
            bucket_name = connection['config'].bucket_name
            
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(remote_key)
            
            blob.upload_from_filename(local_path)
            
            logger.info(f"Uploaded {local_path} to GCS: {remote_key}")
            return True
            
        except Exception as e:
            logger.error(f"GCS upload failed: {str(e)}")
            return False
    
    async def download_file(self, provider: str, remote_key: str, 
                           local_path: str) -> bool:
        """Download a file from data lake."""
        try:
            if provider == 's3' and 's3' in self.active_connections:
                return await self._download_from_s3(remote_key, local_path)
            elif provider == 'azure' and 'azure' in self.active_connections:
                return await self._download_from_azure(remote_key, local_path)
            elif provider == 'gcs' and 'gcs' in self.active_connections:
                return await self._download_from_gcs(remote_key, local_path)
            else:
                logger.error(f"Provider {provider} not connected")
                return False
                
        except Exception as e:
            await self.error_handler.handle_error(
                "download_error",
                f"Failed to download file: {str(e)}",
                error_data={'provider': provider, 'file': remote_key, 'error': str(e)}
            )
            return False
    
    async def _download_from_s3(self, remote_key: str, local_path: str) -> bool:
        """Download file from S3."""
        try:
            connection = self.active_connections['s3']
            client = connection['client']
            bucket = connection['config'].bucket_name
            
            client.download_file(bucket, remote_key, local_path)
            logger.info(f"Downloaded {remote_key} from S3 to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"S3 download failed: {str(e)}")
            return False
    
    async def _download_from_azure(self, remote_key: str, local_path: str) -> bool:
        """Download file from Azure Blob Storage."""
        try:
            connection = self.active_connections['azure']
            client = connection['client']
            container = connection['config'].bucket_name
            
            blob_client = client.get_blob_client(
                container=container, blob=remote_key
            )
            
            with open(local_path, 'wb') as data:
                download_stream = blob_client.download_blob()
                data.write(download_stream.readall())
            
            logger.info(f"Downloaded {remote_key} from Azure to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Azure download failed: {str(e)}")
            return False
    
    async def _download_from_gcs(self, remote_key: str, local_path: str) -> bool:
        """Download file from Google Cloud Storage."""
        try:
            connection = self.active_connections['gcs']
            client = connection['client']
            bucket_name = connection['config'].bucket_name
            
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(remote_key)
            
            blob.download_to_filename(local_path)
            
            logger.info(f"Downloaded {remote_key} from GCS to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"GCS download failed: {str(e)}")
            return False
    
    async def list_objects(self, provider: str, prefix: str = "") -> List[DataLakeObject]:
        """List objects in data lake."""
        try:
            if provider == 's3' and 's3' in self.active_connections:
                return await self._list_s3_objects(prefix)
            elif provider == 'azure' and 'azure' in self.active_connections:
                return await self._list_azure_objects(prefix)
            elif provider == 'gcs' and 'gcs' in self.active_connections:
                return await self._list_gcs_objects(prefix)
            else:
                logger.error(f"Provider {provider} not connected")
                return []
                
        except Exception as e:
            await self.error_handler.handle_error(
                "list_objects_error",
                f"Failed to list objects: {str(e)}",
                error_data={'provider': provider, 'prefix': prefix, 'error': str(e)}
            )
            return []
    
    async def _list_s3_objects(self, prefix: str) -> List[DataLakeObject]:
        """List objects in S3."""
        try:
            connection = self.active_connections['s3']
            client = connection['client']
            bucket = connection['config'].bucket_name
            
            response = client.list_objects_v2(
                Bucket=bucket, Prefix=prefix
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append(DataLakeObject(
                    key=obj['Key'],
                    size=obj['Size'],
                    last_modified=obj['LastModified'],
                    etag=obj.get('ETag')
                ))
            
            return objects
            
        except Exception as e:
            logger.error(f"S3 list objects failed: {str(e)}")
            return []
    
    async def _list_azure_objects(self, prefix: str) -> List[DataLakeObject]:
        """List objects in Azure Blob Storage."""
        try:
            connection = self.active_connections['azure']
            client = connection['client']
            container = connection['config'].bucket_name
            
            container_client = client.get_container_client(container)
            blobs = container_client.list_blobs(name_starts_with=prefix)
            
            objects = []
            for blob in blobs:
                objects.append(DataLakeObject(
                    key=blob.name,
                    size=blob.size,
                    last_modified=blob.last_modified,
                    etag=blob.etag
                ))
            
            return objects
            
        except Exception as e:
            logger.error(f"Azure list objects failed: {str(e)}")
            return []
    
    async def _list_gcs_objects(self, prefix: str) -> List[DataLakeObject]:
        """List objects in Google Cloud Storage."""
        try:
            connection = self.active_connections['gcs']
            client = connection['client']
            bucket_name = connection['config'].bucket_name
            
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            objects = []
            for blob in blobs:
                objects.append(DataLakeObject(
                    key=blob.name,
                    size=blob.size,
                    last_modified=blob.updated,
                    etag=blob.etag
                ))
            
            return objects
            
        except Exception as e:
            logger.error(f"GCS list objects failed: {str(e)}")
            return []
    
    async def delete_object(self, provider: str, remote_key: str) -> bool:
        """Delete an object from data lake."""
        try:
            if provider == 's3' and 's3' in self.active_connections:
                return await self._delete_from_s3(remote_key)
            elif provider == 'azure' and 'azure' in self.active_connections:
                return await self._delete_from_azure(remote_key)
            elif provider == 'gcs' and 'gcs' in self.active_connections:
                return await self._delete_from_gcs(remote_key)
            else:
                logger.error(f"Provider {provider} not connected")
                return False
                
        except Exception as e:
            await self.error_handler.handle_error(
                "delete_object_error",
                f"Failed to delete object: {str(e)}",
                error_data={'provider': provider, 'key': remote_key, 'error': str(e)}
            )
            return False
    
    async def _delete_from_s3(self, remote_key: str) -> bool:
        """Delete object from S3."""
        try:
            connection = self.active_connections['s3']
            client = connection['client']
            bucket = connection['config'].bucket_name
            
            client.delete_object(Bucket=bucket, Key=remote_key)
            logger.info(f"Deleted {remote_key} from S3")
            return True
            
        except Exception as e:
            logger.error(f"S3 delete failed: {str(e)}")
            return False
    
    async def _delete_from_azure(self, remote_key: str) -> bool:
        """Delete object from Azure Blob Storage."""
        try:
            connection = self.active_connections['azure']
            client = connection['client']
            container = connection['config'].bucket_name
            
            blob_client = client.get_blob_client(
                container=container, blob=remote_key
            )
            
            blob_client.delete_blob()
            logger.info(f"Deleted {remote_key} from Azure")
            return True
            
        except Exception as e:
            logger.error(f"Azure delete failed: {str(e)}")
            return False
    
    async def _delete_from_gcs(self, remote_key: str) -> bool:
        """Delete object from Google Cloud Storage."""
        try:
            connection = self.active_connections['gcs']
            client = connection['client']
            bucket_name = connection['config'].bucket_name
            
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(remote_key)
            
            blob.delete()
            logger.info(f"Deleted {remote_key} from GCS")
            return True
            
        except Exception as e:
            logger.error(f"GCS delete failed: {str(e)}")
            return False
    
    async def get_connection_status(self) -> Dict[str, bool]:
        """Get status of all connections."""
        return {
            's3': 's3' in self.active_connections,
            'azure': 'azure' in self.active_connections,
            'gcs': 'gcs' in self.active_connections
        }
