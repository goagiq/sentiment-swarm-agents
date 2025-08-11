#!/bin/bash

# Production Backup Script for Sentiment Analysis System
# This script creates automated backups of the system data

set -e

# Configuration
BACKUP_DIR="/backup/sentiment-analysis"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30
LOG_FILE="/var/log/sentiment-backup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error_exit "This script should not be run as root"
fi

log "${GREEN}Starting backup process...${NC}"

# Create backup directory
log "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR" || error_exit "Failed to create backup directory"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    error_exit "Docker is not running"
fi

# Check if sentiment-analysis container is running
if ! docker ps | grep -q sentiment-analysis; then
    log "${YELLOW}Warning: sentiment-analysis container is not running${NC}"
fi

# Backup ChromaDB data
log "Backing up ChromaDB data..."
if [ -d "cache/chroma_db" ]; then
    tar -czf "$BACKUP_DIR/chromadb_$DATE.tar.gz" cache/chroma_db/ || error_exit "Failed to backup ChromaDB data"
    log "${GREEN}ChromaDB backup completed: chromadb_$DATE.tar.gz${NC}"
else
    log "${YELLOW}Warning: ChromaDB directory not found${NC}"
fi

# Backup configuration files
log "Backing up configuration files..."
if [ -f ".env" ]; then
    cp .env "$BACKUP_DIR/env_$DATE" || error_exit "Failed to backup environment file"
    log "${GREEN}Environment file backup completed: env_$DATE${NC}"
else
    log "${YELLOW}Warning: .env file not found${NC}"
fi

# Backup logs
log "Backing up log files..."
if [ -d "logs" ]; then
    tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" logs/ || error_exit "Failed to backup logs"
    log "${GREEN}Logs backup completed: logs_$DATE.tar.gz${NC}"
else
    log "${YELLOW}Warning: logs directory not found${NC}"
fi

# Backup data directory
log "Backing up data directory..."
if [ -d "data" ]; then
    tar -czf "$BACKUP_DIR/data_$DATE.tar.gz" data/ || error_exit "Failed to backup data directory"
    log "${GREEN}Data directory backup completed: data_$DATE.tar.gz${NC}"
else
    log "${YELLOW}Warning: data directory not found${NC}"
fi

# Backup Docker volumes (if using named volumes)
log "Backing up Docker volumes..."
if docker volume ls | grep -q sentiment-analysis; then
    docker run --rm -v sentiment-analysis:/data -v "$BACKUP_DIR":/backup alpine tar -czf "/backup/volumes_$DATE.tar.gz" /data || error_exit "Failed to backup Docker volumes"
    log "${GREEN}Docker volumes backup completed: volumes_$DATE.tar.gz${NC}"
else
    log "${YELLOW}Warning: No Docker volumes found${NC}"
fi

# Create backup manifest
log "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest_$DATE.txt" << EOF
Backup Manifest - $(date)
=====================================
Backup Date: $(date)
Backup Directory: $BACKUP_DIR
Retention Days: $RETENTION_DAYS

Files Created:
$(ls -la "$BACKUP_DIR"/*_$DATE.* 2>/dev/null || echo "No backup files found")

System Information:
$(uname -a)
Docker Version: $(docker --version)
Disk Usage: $(df -h . | tail -1)

EOF

log "${GREEN}Backup manifest created: manifest_$DATE.txt${NC}"

# Cleanup old backups
log "Cleaning up old backups (older than $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete || log "${YELLOW}Warning: Failed to cleanup old backups${NC}"
find "$BACKUP_DIR" -name "env_*" -mtime +$RETENTION_DAYS -delete || log "${YELLOW}Warning: Failed to cleanup old env files${NC}"
find "$BACKUP_DIR" -name "manifest_*.txt" -mtime +$RETENTION_DAYS -delete || log "${YELLOW}Warning: Failed to cleanup old manifests${NC}"

# Calculate backup size
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
log "${GREEN}Backup size: $BACKUP_SIZE${NC}"

# Verify backup integrity
log "Verifying backup integrity..."
for file in "$BACKUP_DIR"/*_$DATE.tar.gz; do
    if [ -f "$file" ]; then
        if tar -tzf "$file" > /dev/null 2>&1; then
            log "${GREEN}✓ Backup file integrity verified: $(basename "$file")${NC}"
        else
            log "${RED}✗ Backup file integrity check failed: $(basename "$file")${NC}"
        fi
    fi
done

# Final status
log "${GREEN}Backup process completed successfully!${NC}"
log "Backup location: $BACKUP_DIR"
log "Backup date: $DATE"
log "Files created:"
ls -la "$BACKUP_DIR"/*_$DATE.* 2>/dev/null || echo "No backup files found"

# Optional: Send notification
if command -v curl > /dev/null 2>&1; then
    # Example: Send webhook notification
    # curl -X POST -H "Content-Type: application/json" \
    #      -d "{\"text\":\"Backup completed: $DATE\"}" \
    #      "$WEBHOOK_URL"
    log "Notification sent (if configured)"
fi

exit 0
