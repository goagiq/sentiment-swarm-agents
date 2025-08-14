# Deployment Guides - Sentiment Analysis & Decision Support System

**Version:** 1.0.0  
**Last Updated:** 2025-08-14

## Table of Contents

1. [Production Deployment Process](#production-deployment-process)
2. [Environment Setup Guide](#environment-setup-guide)
3. [Monitoring and Alerting Setup](#monitoring-and-alerting-setup)
4. [Backup and Recovery Procedures](#backup-and-recovery-procedures)
5. [Scaling Procedures](#scaling-procedures)
6. [Security Configuration](#security-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting Deployment Issues](#troubleshooting-deployment-issues)

## Production Deployment Process

### Overview

This guide covers the complete production deployment process for the Sentiment Analysis & Decision Support System, including Docker containerization, Kubernetes orchestration, and monitoring setup.

### Prerequisites

- **Infrastructure**: Kubernetes cluster (v1.20+)
- **Storage**: Persistent storage for databases and logs
- **Networking**: Load balancer and ingress controller
- **Monitoring**: Prometheus and Grafana setup
- **Security**: SSL certificates and secrets management

### Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Ingress       │    │   FastAPI       │
│                 │───▶│   Controller    │───▶│   Application   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │   Grafana       │    │   Redis Cache   │
│   Monitoring    │◀───│   Dashboard     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   ChromaDB      │    │   Ollama        │
│   Database      │    │   Vector DB     │    │   Models        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Step-by-Step Deployment

#### 1. Prepare Environment

```bash
# Clone repository
git clone https://github.com/your-org/sentiment-analysis-system.git
cd sentiment-analysis-system

# Set environment variables
export KUBECONFIG=/path/to/your/kubeconfig
export NAMESPACE=sentiment-analysis
export DOMAIN=your-domain.com
```

#### 2. Create Namespace

```bash
# Create namespace
kubectl create namespace $NAMESPACE

# Set namespace as default
kubectl config set-context --current --namespace=$NAMESPACE
```

#### 3. Configure Secrets

```bash
# Create secrets file
cat << EOF > k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: sentiment-analysis-secrets
type: Opaque
data:
  # Base64 encoded values
  database-url: $(echo -n "postgresql://user:password@postgres:5432/sentiment" | base64)
  redis-url: $(echo -n "redis://redis:6379" | base64)
  api-key: $(echo -n "your-api-key" | base64)
  ollama-url: $(echo -n "http://ollama:11434" | base64)
EOF

# Apply secrets
kubectl apply -f k8s/secret.yaml
```

#### 4. Deploy Database

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/persistent-volume.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/postgres-service.yaml

# Wait for database to be ready
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
```

#### 5. Deploy Redis

```bash
# Deploy Redis
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/redis-service.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s
```

#### 6. Deploy Ollama

```bash
# Deploy Ollama with models
kubectl apply -f k8s/ollama-deployment.yaml
kubectl apply -f k8s/ollama-service.yaml

# Wait for Ollama to be ready
kubectl wait --for=condition=ready pod -l app=ollama --timeout=600s

# Pull required models
kubectl exec -it deployment/ollama -- ollama pull llama3.2:latest
kubectl exec -it deployment/ollama -- ollama pull mistral-small3.1:latest
```

#### 7. Deploy Application

```bash
# Build and push Docker image
docker build -t your-registry/sentiment-analysis:latest .
docker push your-registry/sentiment-analysis:latest

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for application to be ready
kubectl wait --for=condition=ready pod -l app=sentiment-analysis --timeout=300s
```

#### 8. Configure Ingress

```bash
# Deploy ingress
kubectl apply -f k8s/ingress.yaml

# Verify ingress
kubectl get ingress
```

#### 9. Deploy Monitoring

```bash
# Deploy Prometheus
kubectl apply -f monitoring/prometheus.yml

# Deploy Grafana
kubectl apply -f monitoring/grafana-datasource.yml
kubectl apply -f monitoring/grafana-dashboard.json

# Deploy monitoring stack
kubectl apply -f k8s/monitoring/
```

#### 10. Verify Deployment

```bash
# Check all pods
kubectl get pods

# Check services
kubectl get services

# Test application
curl -k https://$DOMAIN/health

# Check logs
kubectl logs -l app=sentiment-analysis
```

### Deployment Verification Checklist

- [ ] All pods are running and healthy
- [ ] Services are accessible
- [ ] Ingress is configured correctly
- [ ] SSL certificates are valid
- [ ] Database connections are working
- [ ] Redis cache is operational
- [ ] Ollama models are loaded
- [ ] Monitoring is collecting metrics
- [ ] Logs are being generated
- [ ] Health checks are passing

## Environment Setup Guide

### Development Environment

#### 1. Local Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/sentiment-analysis-system.git
cd sentiment-analysis-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements.dev.txt
```

#### 2. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit environment variables
nano .env
```

**Environment Variables:**

```bash
# Application Settings
APP_NAME=Sentiment Analysis System
APP_VERSION=1.0.0
DEBUG=False
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/sentiment
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_POOL_SIZE=10

# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODELS=llama3.2:latest,mistral-small3.1:latest

# API Configuration
API_HOST=0.0.0.0
API_PORT=8003
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
JWT_SECRET=your-jwt-secret-here

# Monitoring
PROMETHEUS_ENABLED=True
GRAFANA_ENABLED=True
METRICS_PORT=9090

# Logging
LOG_FILE=logs/app.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5
```

#### 3. Database Setup

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib  # Ubuntu
# or
brew install postgresql  # macOS

# Create database
sudo -u postgres createdb sentiment

# Run migrations
python -m src.core.database.migrations

# Seed initial data
python -m src.core.database.seed
```

#### 4. Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server  # Ubuntu
# or
brew install redis  # macOS

# Start Redis
sudo systemctl start redis-server

# Test Redis connection
redis-cli ping
```

#### 5. Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull models
ollama pull llama3.2:latest
ollama pull mistral-small3.1:latest

# Verify models
ollama list
```

### Staging Environment

#### 1. Staging Infrastructure

```bash
# Create staging namespace
kubectl create namespace sentiment-analysis-staging

# Apply staging configurations
kubectl apply -f k8s/staging/ -n sentiment-analysis-staging
```

#### 2. Staging Configuration

```yaml
# k8s/staging/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: staging-config
data:
  APP_ENV: staging
  DEBUG: "True"
  LOG_LEVEL: DEBUG
  DATABASE_URL: postgresql://user:password@postgres-staging:5432/sentiment_staging
  REDIS_URL: redis://redis-staging:6379
  OLLAMA_URL: http://ollama-staging:11434
```

### Production Environment

#### 1. Production Infrastructure

```bash
# Create production namespace
kubectl create namespace sentiment-analysis-prod

# Apply production configurations
kubectl apply -f k8s/production/ -n sentiment-analysis-prod
```

#### 2. Production Configuration

```yaml
# k8s/production/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: production-config
data:
  APP_ENV: production
  DEBUG: "False"
  LOG_LEVEL: WARNING
  DATABASE_URL: postgresql://user:password@postgres-prod:5432/sentiment_prod
  REDIS_URL: redis://redis-prod:6379
  OLLAMA_URL: http://ollama-prod:11434
  SECURITY_HEADERS: "True"
  RATE_LIMITING: "True"
  CORS_ORIGINS: "https://your-domain.com"
```

## Monitoring and Alerting Setup

### Prometheus Configuration

#### 1. Prometheus Setup

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'sentiment-analysis'
    static_configs:
      - targets: ['sentiment-analysis:8003']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

#### 2. Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: sentiment-analysis
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: DatabaseConnectionFailing
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failing"
          description: "PostgreSQL database is not responding"

      - alert: RedisConnectionFailing
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis connection failing"
          description: "Redis cache is not responding"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"
```

### Grafana Dashboards

#### 1. Application Dashboard

```json
{
  "dashboard": {
    "title": "Sentiment Analysis System",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      },
      {
        "title": "Active Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "http_requests_in_progress",
            "legendFormat": "Active requests"
          }
        ]
      }
    ]
  }
}
```

#### 2. System Dashboard

```json
{
  "dashboard": {
    "title": "System Metrics",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU %"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "Memory %"
          }
        ]
      },
      {
        "title": "Disk Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes * 100",
            "legendFormat": "Disk %"
          }
        ]
      }
    ]
  }
}
```

### Alerting Configuration

#### 1. AlertManager Setup

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@your-domain.com'
  smtp_auth_username: 'your-email@gmail.com'
  smtp_auth_password: 'your-app-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'team-sentiment'

receivers:
  - name: 'team-sentiment'
    email_configs:
      - to: 'team@your-domain.com'
        send_resolved: true
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts'
        send_resolved: true
```

#### 2. Slack Integration

```yaml
# Slack webhook configuration
slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#alerts'
    title: '{{ template "slack.title" . }}'
    text: '{{ template "slack.text" . }}'
    send_resolved: true
```

## Backup and Recovery Procedures

### Database Backup

#### 1. Automated Backup Script

```bash
#!/bin/bash
# scripts/backup_database.sh

# Configuration
BACKUP_DIR="/backups/database"
RETENTION_DAYS=30
DATABASE_NAME="sentiment"
POSTGRES_HOST="postgres"
POSTGRES_USER="postgres"

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate backup filename
BACKUP_FILE="$BACKUP_DIR/sentiment_$(date +%Y%m%d_%H%M%S).sql"

# Create backup
kubectl exec -it deployment/postgres -- pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER $DATABASE_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Remove old backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

#### 2. Cron Job for Automated Backups

```yaml
# k8s/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:13
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres -U postgres sentiment | gzip > /backups/sentiment_$(date +%Y%m%d_%H%M%S).sql.gz
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### Application Data Backup

#### 1. Configuration Backup

```bash
#!/bin/bash
# scripts/backup_config.sh

# Backup configuration files
CONFIG_BACKUP_DIR="/backups/config"
mkdir -p $CONFIG_BACKUP_DIR

# Backup Kubernetes configurations
kubectl get all -o yaml > $CONFIG_BACKUP_DIR/k8s_resources_$(date +%Y%m%d_%H%M%S).yaml

# Backup secrets (encrypted)
kubectl get secrets -o yaml > $CONFIG_BACKUP_DIR/secrets_$(date +%Y%m%d_%H%M%S).yaml

# Backup configmaps
kubectl get configmaps -o yaml > $CONFIG_BACKUP_DIR/configmaps_$(date +%Y%m%d_%H%M%S).yaml

echo "Configuration backup completed"
```

#### 2. Log Backup

```bash
#!/bin/bash
# scripts/backup_logs.sh

# Backup application logs
LOG_BACKUP_DIR="/backups/logs"
mkdir -p $LOG_BACKUP_DIR

# Get logs from all pods
kubectl logs -l app=sentiment-analysis --all-containers=true > $LOG_BACKUP_DIR/app_logs_$(date +%Y%m%d_%H%M%S).log

# Compress logs
gzip $LOG_BACKUP_DIR/app_logs_*.log

echo "Log backup completed"
```

### Recovery Procedures

#### 1. Database Recovery

```bash
#!/bin/bash
# scripts/restore_database.sh

# Configuration
BACKUP_FILE=$1
DATABASE_NAME="sentiment"
POSTGRES_HOST="postgres"
POSTGRES_USER="postgres"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
kubectl scale deployment sentiment-analysis --replicas=0

# Wait for pods to stop
kubectl wait --for=delete pod -l app=sentiment-analysis --timeout=300s

# Drop and recreate database
kubectl exec -it deployment/postgres -- psql -h $POSTGRES_HOST -U $POSTGRES_USER -c "DROP DATABASE IF EXISTS $DATABASE_NAME;"
kubectl exec -it deployment/postgres -- psql -h $POSTGRES_HOST -U $POSTGRES_USER -c "CREATE DATABASE $DATABASE_NAME;"

# Restore from backup
gunzip -c $BACKUP_FILE | kubectl exec -i deployment/postgres -- psql -h $POSTGRES_HOST -U $POSTGRES_USER $DATABASE_NAME

# Restart application
kubectl scale deployment sentiment-analysis --replicas=3

echo "Database recovery completed"
```

#### 2. Full System Recovery

```bash
#!/bin/bash
# scripts/full_recovery.sh

# Full system recovery procedure
echo "Starting full system recovery..."

# 1. Restore Kubernetes resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmaps.yaml

# 2. Restore databases
./scripts/restore_database.sh /backups/database/latest_backup.sql.gz

# 3. Restore application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# 4. Verify recovery
kubectl wait --for=condition=ready pod -l app=sentiment-analysis --timeout=300s
curl -k https://your-domain.com/health

echo "Full system recovery completed"
```

## Scaling Procedures

### Horizontal Scaling

#### 1. Application Scaling

```bash
# Scale application pods
kubectl scale deployment sentiment-analysis --replicas=5

# Check scaling status
kubectl get pods -l app=sentiment-analysis

# Monitor scaling
kubectl top pods -l app=sentiment-analysis
```

#### 2. Auto-scaling Configuration

```yaml
# k8s/horizontal-pod-autoscaler.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-analysis-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-analysis
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Database Scaling

#### 1. Read Replicas

```yaml
# k8s/postgres-read-replica.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-read-replica
spec:
  replicas: 2
  selector:
    matchLabels:
      app: postgres-read-replica
  template:
    metadata:
      labels:
        app: postgres-read-replica
    spec:
      containers:
      - name: postgres
        image: postgres:13
        env:
        - name: POSTGRES_DB
          value: sentiment
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-read-replica
spec:
  selector:
    app: postgres-read-replica
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

#### 2. Connection Pooling

```yaml
# k8s/pgpool-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgpool
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgpool
  template:
    metadata:
      labels:
        app: pgpool
    spec:
      containers:
      - name: pgpool
        image: bitnami/pgpool:latest
        env:
        - name: POSTGRESQL_HOST
          value: postgres
        - name: POSTGRESQL_PORT_NUMBER
          value: "5432"
        - name: POSTGRESQL_USERNAME
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRESQL_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
```

### Cache Scaling

#### 1. Redis Cluster

```yaml
# k8s/redis-cluster.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:6-alpine
        command: ["redis-server", "/etc/redis/redis.conf"]
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-config
          mountPath: /etc/redis
        - name: redis-data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

### Load Balancing

#### 1. Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sentiment-analysis-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: tls-secret
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sentiment-analysis
            port:
              number: 8003
```

#### 2. Service Mesh (Optional)

```yaml
# k8s/istio-virtualservice.yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: sentiment-analysis
spec:
  hosts:
  - your-domain.com
  gateways:
  - sentiment-analysis-gateway
  http:
  - route:
    - destination:
        host: sentiment-analysis
        port:
          number: 8003
      weight: 80
    - destination:
        host: sentiment-analysis-v2
        port:
          number: 8003
      weight: 20
    retries:
      attempts: 3
      perTryTimeout: 2s
    timeout: 10s
```

## Security Configuration

### SSL/TLS Setup

#### 1. Certificate Management

```bash
# Generate SSL certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=your-domain.com"

# Create Kubernetes secret
kubectl create secret tls tls-secret --key tls.key --cert tls.crt
```

#### 2. Let's Encrypt Integration

```yaml
# k8s/cert-manager.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sentiment-analysis-network-policy
spec:
  podSelector:
    matchLabels:
      app: sentiment-analysis
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8003
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### RBAC Configuration

```yaml
# k8s/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ServiceAccount
metadata:
  name: sentiment-analysis-sa
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sentiment-analysis-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sentiment-analysis-rolebinding
subjects:
- kind: ServiceAccount
  name: sentiment-analysis-sa
roleRef:
  kind: Role
  name: sentiment-analysis-role
  apiGroup: rbac.authorization.k8s.io
```

## Performance Tuning

### Application Tuning

#### 1. Resource Limits

```yaml
# k8s/deployment.yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

#### 2. JVM Tuning (if applicable)

```yaml
env:
- name: JAVA_OPTS
  value: "-Xms512m -Xmx1g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

#### 3. Database Tuning

```yaml
# PostgreSQL configuration
postgresql.conf:
  shared_buffers: 256MB
  effective_cache_size: 1GB
  maintenance_work_mem: 64MB
  checkpoint_completion_target: 0.9
  wal_buffers: 16MB
  default_statistics_target: 100
  random_page_cost: 1.1
  effective_io_concurrency: 200
  work_mem: 4MB
  min_wal_size: 1GB
  max_wal_size: 4GB
```

### Monitoring Performance

#### 1. Performance Metrics

```python
# Custom metrics collection
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])

# Business metrics
ANALYSIS_COUNT = Counter('sentiment_analysis_total', 'Total sentiment analyses', ['language', 'model'])
PREDICTION_ACCURACY = Gauge('prediction_accuracy', 'Prediction accuracy', ['model_type'])

# System metrics
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
```

#### 2. Performance Alerts

```yaml
# monitoring/performance_alerts.yml
groups:
  - name: performance
    rules:
      - alert: HighMemoryUsage
        expr: memory_usage_bytes / memory_total_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          
      - alert: SlowQueries
        expr: rate(database_query_duration_seconds_sum[5m]) / rate(database_query_duration_seconds_count[5m]) > 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Slow database queries detected"
```

## Troubleshooting Deployment Issues

### Common Issues

#### 1. Pod Startup Issues

```bash
# Check pod status
kubectl get pods
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
kubectl logs <pod-name> --previous

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

#### 2. Service Connectivity Issues

```bash
# Check service endpoints
kubectl get endpoints
kubectl describe service <service-name>

# Test connectivity
kubectl run test-pod --image=busybox --rm -it --restart=Never -- nslookup <service-name>
```

#### 3. Resource Issues

```bash
# Check resource usage
kubectl top pods
kubectl top nodes

# Check resource limits
kubectl describe pod <pod-name> | grep -A 10 "Limits:"
```

### Debugging Commands

```bash
# Get detailed pod information
kubectl get pod <pod-name> -o yaml

# Execute commands in running pods
kubectl exec -it <pod-name> -- /bin/bash

# Port forward for local debugging
kubectl port-forward <pod-name> 8003:8003

# Check network policies
kubectl get networkpolicies
kubectl describe networkpolicy <policy-name>
```

### Recovery Procedures

#### 1. Pod Recovery

```bash
# Restart deployment
kubectl rollout restart deployment sentiment-analysis

# Check rollout status
kubectl rollout status deployment sentiment-analysis

# Rollback if needed
kubectl rollout undo deployment sentiment-analysis
```

#### 2. Service Recovery

```bash
# Restart service
kubectl delete service sentiment-analysis
kubectl apply -f k8s/service.yaml

# Check service endpoints
kubectl get endpoints sentiment-analysis
```

#### 3. Database Recovery

```bash
# Check database connectivity
kubectl exec -it deployment/postgres -- pg_isready -h localhost

# Restart database if needed
kubectl rollout restart deployment postgres

# Check database logs
kubectl logs deployment/postgres
```

---

**Deployment Guide Version:** 1.0.0  
**Last Updated:** 2025-08-14  
**For Support:** Check troubleshooting section or contact deployment team
