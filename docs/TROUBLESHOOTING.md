# Troubleshooting Guide

This guide provides solutions for common issues encountered when running the Sentiment Analysis System in production.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Service Issues](#service-issues)
3. [Performance Problems](#performance-problems)
4. [Model Issues](#model-issues)
5. [Database Problems](#database-problems)
6. [Network Issues](#network-issues)
7. [Security Issues](#security-issues)
8. [Monitoring Issues](#monitoring-issues)
9. [Emergency Procedures](#emergency-procedures)

## Quick Diagnostics

### Health Check Script

Use the built-in health check script for comprehensive diagnostics:

```bash
# Run full health check
./scripts/health_check.sh

# JSON output for monitoring systems
./scripts/health_check.sh --json

# Quiet mode
./scripts/health_check.sh --quiet
```

### Quick Status Check

```bash
# Check if all services are running
docker-compose -f docker-compose.prod.yml ps

# Check container logs
docker-compose -f docker-compose.prod.yml logs --tail=50

# Check resource usage
docker stats --no-stream
```

### Common Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| 500 | Internal Server Error | Check application logs |
| 503 | Service Unavailable | Check if Ollama is running |
| 401 | Unauthorized | Verify API key |
| 429 | Too Many Requests | Check rate limiting |
| 413 | Payload Too Large | Reduce file size |

## Service Issues

### Sentiment Analysis Container Not Starting

**Symptoms:**
- Container exits immediately
- "Exit code 1" in logs
- Service unavailable

**Diagnosis:**
```bash
# Check container logs
docker logs sentiment-analysis

# Check if ports are available
netstat -tulpn | grep :8002

# Check resource limits
docker stats sentiment-analysis
```

**Solutions:**

1. **Port Conflict:**
   ```bash
   # Stop conflicting service
   sudo lsof -ti:8002 | xargs kill -9
   
   # Or change port in docker-compose.prod.yml
   ports:
     - "8003:8002"  # Use different external port
   ```

2. **Resource Issues:**
   ```bash
   # Increase memory limit
   docker-compose -f docker-compose.prod.yml down
   # Edit docker-compose.prod.yml to increase limits
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Environment Variables:**
   ```bash
   # Check environment file
   cat .env
   
   # Verify required variables
   grep -E "^(TEXT_MODEL|VISION_MODEL|OLLAMA_HOST)=" .env
   ```

### Ollama Service Issues

**Symptoms:**
- "Connection refused" errors
- Model not found errors
- Slow response times

**Diagnosis:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Check available models
ollama list

# Check Ollama logs
docker logs ollama
```

**Solutions:**

1. **Model Not Found:**
   ```bash
   # Pull required models
   ollama pull mistral-small3.1:latest
   ollama pull llava:latest
   ollama pull llama3.2:latest
   ```

2. **Memory Issues:**
   ```bash
   # Check available memory
   free -h
   
   # Use smaller models
   ollama pull phi3:mini
   # Update .env to use smaller models
   ```

3. **Network Issues:**
   ```bash
   # Check network connectivity
   docker exec sentiment-analysis ping ollama
   
   # Verify Ollama host configuration
   echo $OLLAMA_HOST
   ```

### Redis Connection Issues

**Symptoms:**
- Cache errors
- "Connection refused" to Redis
- Performance degradation

**Diagnosis:**
```bash
# Check Redis status
docker exec redis redis-cli ping

# Check Redis logs
docker logs redis

# Check Redis memory usage
docker exec redis redis-cli info memory
```

**Solutions:**

1. **Redis Not Running:**
   ```bash
   # Restart Redis
   docker-compose -f docker-compose.prod.yml restart redis
   
   # Check Redis configuration
   docker exec redis redis-cli config get maxmemory
   ```

2. **Memory Issues:**
   ```bash
   # Clear Redis cache
   docker exec redis redis-cli flushall
   
   # Increase memory limit in docker-compose.prod.yml
   ```

## Performance Problems

### Slow Response Times

**Symptoms:**
- API responses > 5 seconds
- Timeout errors
- High CPU usage

**Diagnosis:**
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8002/health

# Monitor resource usage
htop

# Check for bottlenecks
docker stats --no-stream
```

**Solutions:**

1. **Model Loading:**
   ```bash
   # Use smaller models
   sed -i 's/mistral-small3.1:latest/phi3:mini/' .env
   
   # Preload models
   ollama pull phi3:mini
   ```

2. **Resource Limits:**
   ```bash
   # Increase container resources
   # Edit docker-compose.prod.yml
   deploy:
     resources:
       limits:
         memory: 4G
         cpus: '2.0'
   ```

3. **Caching:**
   ```bash
   # Enable Redis caching
   # Verify REDIS_HOST in .env
   # Check cache hit rates
   docker exec redis redis-cli info stats
   ```

### High Memory Usage

**Symptoms:**
- Out of memory errors
- Container restarts
- Slow performance

**Diagnosis:**
```bash
# Check memory usage
docker stats --no-stream

# Check memory by process
docker exec sentiment-analysis ps aux --sort=-%mem | head -10

# Check for memory leaks
docker exec sentiment-analysis python -c "import gc; gc.collect()"
```

**Solutions:**

1. **Memory Optimization:**
   ```bash
   # Restart with memory cleanup
   docker-compose -f docker-compose.prod.yml restart sentiment-analysis
   
   # Use memory-efficient models
   sed -i 's/llava:latest/llava:7b/' .env
   ```

2. **Garbage Collection:**
   ```python
   # Add to application code
   import gc
   gc.collect()
   ```

3. **Resource Limits:**
   ```yaml
   # Set appropriate limits
   resources:
     limits:
       memory: 2G
     reservations:
       memory: 1G
   ```

### High CPU Usage

**Symptoms:**
- CPU usage > 90%
- Slow response times
- System unresponsive

**Diagnosis:**
```bash
# Check CPU usage
top -p $(docker inspect sentiment-analysis --format='{{.State.Pid}}')

# Check CPU by process
docker exec sentiment-analysis ps aux --sort=-%cpu | head -10
```

**Solutions:**

1. **Optimize Processing:**
   ```bash
   # Reduce worker count
   sed -i 's/MAX_WORKERS=4/MAX_WORKERS=2/' .env
   
   # Use smaller chunk sizes
   sed -i 's/CHUNK_SIZE=1200/CHUNK_SIZE=600/' .env
   ```

2. **Model Optimization:**
   ```bash
   # Use quantized models
   ollama pull mistral-small3.1:latest-q4_K_M
   ```

## Model Issues

### Model Not Found

**Symptoms:**
- "Model not found" errors
- Fallback to rule-based analysis
- Poor sentiment accuracy

**Diagnosis:**
```bash
# Check available models
ollama list

# Check model status
curl http://localhost:11434/api/tags
```

**Solutions:**

1. **Pull Missing Models:**
   ```bash
   # Pull required models
   ollama pull mistral-small3.1:latest
   ollama pull llava:latest
   ollama pull llama3.2:latest
   ollama pull granite3.2-vision
   ```

2. **Update Model Configuration:**
   ```bash
   # Edit .env file
   TEXT_MODEL=ollama:mistral-small3.1:latest
   VISION_MODEL=ollama:llava:latest
   FALLBACK_TEXT_MODEL=ollama:llama3.2:latest
   ```

### Model Performance Issues

**Symptoms:**
- Inconsistent results
- Poor accuracy
- Slow inference

**Solutions:**

1. **Model Selection:**
   ```bash
   # Try different models
   ollama pull llama3.2:latest
   sed -i 's/mistral-small3.1:latest/llama3.2:latest/' .env
   ```

2. **Parameter Tuning:**
   ```bash
   # Adjust temperature
   sed -i 's/TEXT_TEMPERATURE=0.1/TEXT_TEMPERATURE=0.3/' .env
   
   # Adjust max tokens
   sed -i 's/TEXT_MAX_TOKENS=200/TEXT_MAX_TOKENS=100/' .env
   ```

## Database Problems

### ChromaDB Issues

**Symptoms:**
- Vector database errors
- Search failures
- Data corruption

**Diagnosis:**
```bash
# Check ChromaDB directory
ls -la cache/chroma_db/

# Check disk space
df -h cache/

# Check ChromaDB logs
docker logs sentiment-analysis | grep -i chroma
```

**Solutions:**

1. **Data Corruption:**
   ```bash
   # Backup and restore
   tar -czf chromadb_backup_$(date +%Y%m%d).tar.gz cache/chroma_db/
   rm -rf cache/chroma_db/
   mkdir -p cache/chroma_db/
   ```

2. **Disk Space:**
   ```bash
   # Clean up old data
   find cache/chroma_db/ -name "*.parquet" -mtime +30 -delete
   
   # Increase disk space
   # Add more storage to volume
   ```

### Redis Issues

**Symptoms:**
- Cache misses
- Connection errors
- Memory exhaustion

**Diagnosis:**
```bash
# Check Redis status
docker exec redis redis-cli ping

# Check Redis memory
docker exec redis redis-cli info memory

# Check Redis logs
docker logs redis
```

**Solutions:**

1. **Memory Issues:**
   ```bash
   # Clear cache
   docker exec redis redis-cli flushall
   
   # Adjust memory policy
   docker exec redis redis-cli config set maxmemory-policy allkeys-lru
   ```

2. **Connection Issues:**
   ```bash
   # Restart Redis
   docker-compose -f docker-compose.prod.yml restart redis
   
   # Check network
   docker network ls
   docker network inspect sentiment-network
   ```

## Network Issues

### Connectivity Problems

**Symptoms:**
- Timeout errors
- Connection refused
- Slow downloads

**Diagnosis:**
```bash
# Check network connectivity
ping 8.8.8.8

# Check DNS resolution
nslookup google.com

# Check firewall
sudo iptables -L
```

**Solutions:**

1. **Firewall Issues:**
   ```bash
   # Allow required ports
   sudo ufw allow 8002
   sudo ufw allow 8000
   sudo ufw allow 11434
   ```

2. **DNS Issues:**
   ```bash
   # Update DNS servers
   echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf
   echo "nameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf
   ```

### Docker Network Issues

**Symptoms:**
- Container communication failures
- Service discovery issues
- Port binding problems

**Diagnosis:**
```bash
# Check Docker networks
docker network ls

# Check container networking
docker inspect sentiment-analysis | grep -A 10 "NetworkSettings"

# Test inter-container communication
docker exec sentiment-analysis ping ollama
```

**Solutions:**

1. **Network Configuration:**
   ```bash
   # Recreate network
   docker-compose -f docker-compose.prod.yml down
   docker network prune
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Port Conflicts:**
   ```bash
   # Check port usage
   sudo netstat -tulpn | grep :8002
   
   # Change ports if needed
   # Edit docker-compose.prod.yml
   ```

## Security Issues

### API Key Problems

**Symptoms:**
- 401 Unauthorized errors
- Authentication failures
- Security warnings

**Diagnosis:**
```bash
# Check API key configuration
grep API_KEY .env

# Test API key
curl -H "X-API-Key: your-api-key" http://localhost:8002/health
```

**Solutions:**

1. **Invalid API Key:**
   ```bash
   # Generate new API key
   openssl rand -hex 32
   
   # Update .env file
   sed -i 's/API_KEY=.*/API_KEY=new-generated-key/' .env
   ```

2. **Missing API Key:**
   ```bash
   # Add API key to .env
   echo "API_KEY=your-secure-api-key" >> .env
   ```

### SSL/TLS Issues

**Symptoms:**
- Certificate errors
- HTTPS failures
- Security warnings

**Diagnosis:**
```bash
# Check SSL certificate
openssl x509 -in nginx/ssl/cert.pem -text -noout

# Test SSL connection
curl -k https://localhost:443
```

**Solutions:**

1. **Certificate Expired:**
   ```bash
   # Generate new certificate
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/ssl/key.pem \
     -out nginx/ssl/cert.pem
   ```

2. **Certificate Issues:**
   ```bash
   # Use Let's Encrypt for production
   # Install certbot and configure automatic renewal
   ```

## Monitoring Issues

### Prometheus Problems

**Symptoms:**
- Metrics not available
- Dashboard errors
- Missing data

**Diagnosis:**
```bash
# Check Prometheus status
curl http://localhost:9090/-/healthy

# Check metrics endpoint
curl http://localhost:8002/metrics

# Check Prometheus logs
docker logs prometheus
```

**Solutions:**

1. **Metrics Not Available:**
   ```bash
   # Restart Prometheus
   docker-compose -f docker-compose.prod.yml restart prometheus
   
   # Check configuration
   docker exec prometheus cat /etc/prometheus/prometheus.yml
   ```

2. **Target Discovery:**
   ```bash
   # Check targets in Prometheus UI
   # http://localhost:9090/targets
   
   # Verify service discovery
   docker network inspect sentiment-network
   ```

### Grafana Issues

**Symptoms:**
- Dashboard not loading
- No data displayed
- Authentication problems

**Diagnosis:**
```bash
# Check Grafana status
curl http://localhost:3000/api/health

# Check Grafana logs
docker logs grafana

# Check datasource
curl http://localhost:3000/api/datasources
```

**Solutions:**

1. **Dashboard Issues:**
   ```bash
   # Restart Grafana
   docker-compose -f docker-compose.prod.yml restart grafana
   
   # Import dashboard manually
   # Access http://localhost:3000 and import dashboard
   ```

2. **Datasource Issues:**
   ```bash
   # Check Prometheus connection
   curl http://prometheus:9090/api/v1/query?query=up
   
   # Update datasource configuration
   # Edit monitoring/grafana-datasource.yml
   ```

## Emergency Procedures

### Service Outage

**Immediate Actions:**
```bash
# 1. Check service status
docker-compose -f docker-compose.prod.yml ps

# 2. Check logs for errors
docker-compose -f docker-compose.prod.yml logs --tail=100

# 3. Restart services
docker-compose -f docker-compose.prod.yml restart

# 4. If still failing, force restart
docker-compose -f docker-compose.prod.yml down
docker system prune -f
docker-compose -f docker-compose.prod.yml up -d
```

### Data Loss

**Recovery Actions:**
```bash
# 1. Stop services
docker-compose -f docker-compose.prod.yml down

# 2. Restore from backup
./scripts/restore.sh latest_backup.tar.gz

# 3. Verify data integrity
docker exec sentiment-analysis python -c "from src.core.vector_db import VectorDB; db = VectorDB(); print('ChromaDB is healthy')"

# 4. Restart services
docker-compose -f docker-compose.prod.yml up -d
```

### Security Breach

**Immediate Actions:**
```bash
# 1. Rotate API keys
openssl rand -hex 32 > new_api_key.txt
sed -i 's/API_KEY=.*/API_KEY='$(cat new_api_key.txt)'/' .env

# 2. Check access logs
tail -f logs/access.log | grep -i "unauthorized\|forbidden"

# 3. Restart services
docker-compose -f docker-compose.prod.yml restart

# 4. Review security configuration
grep -E "^(API_KEY|CORS_ORIGINS|RATE_LIMIT)=" .env
```

### Performance Crisis

**Immediate Actions:**
```bash
# 1. Scale down processing
sed -i 's/MAX_WORKERS=4/MAX_WORKERS=1/' .env

# 2. Use smaller models
sed -i 's/mistral-small3.1:latest/phi3:mini/' .env

# 3. Clear caches
docker exec redis redis-cli flushall

# 4. Restart with reduced load
docker-compose -f docker-compose.prod.yml restart
```

## Getting Help

### Logs and Debugging

**Collect Debug Information:**
```bash
# Create debug package
mkdir debug_$(date +%Y%m%d_%H%M%S)
cd debug_$(date +%Y%m%d_%H%M%S)

# Collect logs
docker-compose -f ../docker-compose.prod.yml logs > docker_logs.txt
docker stats --no-stream > docker_stats.txt
./scripts/health_check.sh > health_check.txt

# Collect system info
uname -a > system_info.txt
df -h > disk_usage.txt
free -h > memory_usage.txt

# Create archive
tar -czf ../debug_package.tar.gz .
```

### Support Resources

- **Documentation**: Check the main README and docs/ directory
- **GitHub Issues**: Report bugs and feature requests
- **Community**: Join discussions and get help
- **Monitoring**: Use Grafana dashboards for insights

### Contact Information

- **System Administrator**: admin@your-domain.com
- **DevOps Team**: devops@your-domain.com
- **Security Team**: security@your-domain.com
- **Emergency**: +1-555-0123 (24/7 support)
