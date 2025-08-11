# Production Deployment Guide

This guide provides comprehensive instructions for deploying the Sentiment Analysis System in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring Setup](#monitoring-setup)
6. [Security Configuration](#security-configuration)
7. [Performance Optimization](#performance-optimization)
8. [Backup and Recovery](#backup-and-recovery)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 4+ cores (8+ recommended for production)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 50GB+ available space
- **OS**: Linux (Ubuntu 20.04+ recommended) or Windows Server 2019+
- **Network**: Stable internet connection for model downloads

### Software Requirements

- **Docker**: 20.10+ with Docker Compose
- **Kubernetes**: 1.24+ (if using K8s deployment)
- **Ollama**: Latest version
- **FFmpeg**: For video processing
- **Redis**: 7.0+ (for caching)

### Security Requirements

- **SSL Certificate**: Valid SSL certificate for HTTPS
- **Firewall**: Configured to allow only necessary ports
- **API Keys**: Secure API keys for external services
- **Secrets Management**: Proper secrets management system

## Environment Setup

### 1. Clone and Prepare Repository

```bash
# Clone the repository
git clone <repository-url>
cd Sentiment

# Create production environment file
cp env.production .env

# Edit environment variables
nano .env
```

### 2. Configure Environment Variables

Update the `.env` file with your production settings:

```bash
# Essential settings
NODE_ENV=production
API_KEY=your-secure-api-key
CORS_ORIGINS=https://your-domain.com

# Ollama configuration
OLLAMA_HOST=http://ollama:11434
TEXT_MODEL=ollama:mistral-small3.1:latest
VISION_MODEL=ollama:llava:latest

# Monitoring
ENABLE_METRICS=true
SENTRY_DSN=your-sentry-dsn

# Storage
CHROMA_PERSIST_DIRECTORY=/app/data/chroma
CACHE_DIRECTORY=/app/cache
```

### 3. Create Required Directories

```bash
# Create data directories
mkdir -p data cache logs temp

# Set proper permissions
chmod 755 data cache logs temp
```

## Docker Deployment

### 1. Single Container Deployment

```bash
# Build the production image
docker build -t sentiment-analysis:latest .

# Run with production settings
docker run -d \
  --name sentiment-analysis \
  --restart unless-stopped \
  -p 8000:8000 \
  -p 8002:8002 \
  -e NODE_ENV=production \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/logs:/app/logs \
  sentiment-analysis:latest
```

### 2. Multi-Container Deployment

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f sentiment-analysis
```

### 3. Production Docker Compose

The `docker-compose.prod.yml` includes:

- **Sentiment Analysis**: Main application
- **Ollama**: LLM server
- **Redis**: Caching layer
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboard
- **Nginx**: Reverse proxy and load balancer

### 4. SSL Configuration

```bash
# Generate SSL certificates (for development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem

# For production, use Let's Encrypt or your CA
```

## Kubernetes Deployment

### 1. Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Verify installation
kubectl version --client
```

### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply persistent volumes
kubectl apply -f k8s/persistent-volume.yaml

# Apply configuration
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Deploy services
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment.yaml

# Apply ingress (if using)
kubectl apply -f k8s/ingress.yaml
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n sentiment-analysis

# Check services
kubectl get svc -n sentiment-analysis

# Check ingress
kubectl get ingress -n sentiment-analysis

# View logs
kubectl logs -f deployment/sentiment-analysis -n sentiment-analysis
```

### 4. Scale Deployment

```bash
# Scale to 5 replicas
kubectl scale deployment sentiment-analysis --replicas=5 -n sentiment-analysis

# Check scaling status
kubectl get pods -n sentiment-analysis
```

## Monitoring Setup

### 1. Prometheus Configuration

The system automatically exposes metrics at `/metrics` endpoint:

```bash
# Test metrics endpoint
curl http://localhost:8002/metrics

# Example metrics
sentiment_analysis_requests_total{endpoint="/analyze_text",status="200"} 1234
sentiment_analysis_duration_seconds{endpoint="/analyze_text"} 0.5
sentiment_analysis_model_usage{model="mistral-small3.1"} 567
```

### 2. Grafana Dashboard

Access Grafana at `http://localhost:3000`:

- **Username**: admin
- **Password**: admin (change on first login)

Import the provided dashboard configuration:

```bash
# Import dashboard
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @monitoring/grafana-dashboard.json
```

### 3. Alerting Configuration

Create alert rules in Prometheus:

```yaml
# monitoring/alerts.yml
groups:
  - name: sentiment-analysis
    rules:
      - alert: HighErrorRate
        expr: rate(sentiment_analysis_requests_total{status=~"4..|5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: High error rate detected
          description: Error rate is {{ $value }} errors per second

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(sentiment_analysis_duration_seconds_bucket[5m])) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: High response time detected
          description: 95th percentile response time is {{ $value }} seconds
```

## Security Configuration

### 1. API Security

```python
# Enable API key authentication
API_KEY_HEADER = "X-API-Key"
API_KEY = os.getenv("API_KEY")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path in ["/health", "/metrics", "/docs"]:
        return await call_next(request)
    
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return await call_next(request)
```

### 2. CORS Configuration

```python
# Configure CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 3. Rate Limiting

```python
# Implement rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/analyze_text")
@limiter.limit("100/minute")
async def analyze_text(request: Request, text: str):
    # Implementation
    pass
```

### 4. SSL/TLS Configuration

```nginx
# nginx/nginx.conf
server {
    listen 443 ssl http2;
    server_name sentiment.your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
```

## Performance Optimization

### 1. Caching Strategy

```python
# Redis caching for expensive operations
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expire_time=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached_result = redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expire_time, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### 2. Connection Pooling

```python
# Optimize database connections
import aiohttp

# Create connection pool
conn = aiohttp.TCPConnector(
    limit=100,
    limit_per_host=30,
    ttl_dns_cache=300,
    use_dns_cache=True
)

session = aiohttp.ClientSession(connector=conn)
```

### 3. Memory Management

```python
# Implement memory-efficient processing
import gc
import psutil

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    
    if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
        gc.collect()
        return True
    return False
```

### 4. Resource Limits

```yaml
# k8s/deployment.yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

## Backup and Recovery

### 1. Automated Backup Script

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backup/sentiment-analysis"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup ChromaDB data
tar -czf $BACKUP_DIR/chromadb_$DATE.tar.gz cache/chroma_db/

# Backup configuration
cp .env $BACKUP_DIR/env_$DATE

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR"
```

### 2. Recovery Process

```bash
#!/bin/bash
# scripts/restore.sh

BACKUP_FILE=$1
BACKUP_DIR="/backup/sentiment-analysis"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore data
tar -xzf $BACKUP_DIR/$BACKUP_FILE -C /

# Restore configuration
cp $BACKUP_DIR/env_$(echo $BACKUP_FILE | cut -d'_' -f2) .env

# Start services
docker-compose -f docker-compose.prod.yml up -d

echo "Recovery completed"
```

### 3. Database Backup

```bash
# Backup ChromaDB
docker exec sentiment-analysis tar -czf /app/backup/chromadb_$(date +%Y%m%d_%H%M%S).tar.gz /app/cache/chroma_db/

# Backup Redis
docker exec redis redis-cli BGSAVE
docker cp redis:/data/dump.rdb ./backup/redis_$(date +%Y%m%d_%H%M%S).rdb
```

## Troubleshooting

### 1. Common Issues

#### Service Not Starting
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs sentiment-analysis

# Check resource usage
docker stats

# Check port conflicts
netstat -tulpn | grep :8002
```

#### High Memory Usage
```bash
# Monitor memory usage
docker stats --no-stream

# Check for memory leaks
docker exec sentiment-analysis ps aux --sort=-%mem | head -10

# Restart service if needed
docker-compose -f docker-compose.prod.yml restart sentiment-analysis
```

#### Slow Response Times
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Check Redis performance
docker exec redis redis-cli info memory

# Check network connectivity
docker exec sentiment-analysis ping ollama
```

### 2. Health Checks

```bash
#!/bin/bash
# scripts/health_check.sh

HEALTH_URL="http://localhost:8002/health"
METRICS_URL="http://localhost:8002/metrics"

# Check API health
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)
if [ $RESPONSE -eq 200 ]; then
    echo "API is healthy"
else
    echo "API is unhealthy (HTTP $RESPONSE)"
    exit 1
fi

# Check metrics endpoint
METRICS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $METRICS_URL)
if [ $METRICS_RESPONSE -eq 200 ]; then
    echo "Metrics endpoint is healthy"
else
    echo "Metrics endpoint is unhealthy (HTTP $METRICS_RESPONSE)"
    exit 1
fi

echo "All health checks passed"
```

### 3. Performance Monitoring

```bash
# Monitor CPU and memory
htop

# Monitor disk usage
df -h

# Monitor network
iftop

# Monitor logs
tail -f logs/sentiment.log | grep ERROR
```

### 4. Emergency Procedures

#### Service Outage
```bash
# Quick restart
docker-compose -f docker-compose.prod.yml restart

# Force restart with cleanup
docker-compose -f docker-compose.prod.yml down
docker system prune -f
docker-compose -f docker-compose.prod.yml up -d
```

#### Data Corruption
```bash
# Restore from latest backup
./scripts/restore.sh chromadb_20231201_120000.tar.gz

# Verify data integrity
docker exec sentiment-analysis python -c "from src.core.vector_db import VectorDB; db = VectorDB(); print('ChromaDB is healthy')"
```

#### Security Incident
```bash
# Rotate API keys
sed -i 's/API_KEY=.*/API_KEY=new-secure-key/' .env

# Restart services
docker-compose -f docker-compose.prod.yml restart

# Check access logs
tail -f logs/access.log | grep -i "unauthorized\|forbidden"
```

## Maintenance Schedule

### Daily Tasks
- [ ] Check service health
- [ ] Monitor error logs
- [ ] Verify backup completion
- [ ] Check resource usage

### Weekly Tasks
- [ ] Review performance metrics
- [ ] Update security patches
- [ ] Clean up temporary files
- [ ] Verify SSL certificate validity

### Monthly Tasks
- [ ] Review and update dependencies
- [ ] Analyze usage patterns
- [ ] Update monitoring dashboards
- [ ] Test disaster recovery procedures

### Quarterly Tasks
- [ ] Security audit
- [ ] Performance optimization review
- [ ] Capacity planning
- [ ] Documentation updates

## Support and Resources

### Documentation
- [Main README](../README.md)
- [API Documentation](http://localhost:8002/docs)
- [Configuration Guide](../docs/CONFIGURABLE_MODELS_GUIDE.md)

### Monitoring Dashboards
- [Grafana Dashboard](http://localhost:3000)
- [Prometheus Metrics](http://localhost:9090)

### Logs and Debugging
- Application logs: `logs/sentiment.log`
- Docker logs: `docker-compose logs -f`
- Kubernetes logs: `kubectl logs -f deployment/sentiment-analysis`

### Emergency Contacts
- System Administrator: admin@your-domain.com
- DevOps Team: devops@your-domain.com
- Security Team: security@your-domain.com
