#!/bin/bash

# Health Check Script for Sentiment Analysis System
# This script performs comprehensive health checks on the system

set -e

# Configuration
HEALTH_URL="http://localhost:8002/health"
METRICS_URL="http://localhost:8002/metrics"
MCP_URL="http://localhost:8000/mcp"
ADVANCED_ANALYTICS_URL="http://localhost:8003/advanced-analytics/health"
MAIN_API_URL="http://localhost:8003/health"
OLLAMA_URL="http://localhost:11434/api/tags"
REDIS_HOST="localhost"
REDIS_PORT="6379"
LOG_FILE="/var/log/sentiment-health.log"

# Thresholds
MAX_RESPONSE_TIME=5
MAX_MEMORY_USAGE=80
MAX_DISK_USAGE=85
MAX_CPU_USAGE=90

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Status tracking
HEALTH_STATUS=0
ERRORS=()

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Check function
check() {
    local name="$1"
    local command="$2"
    local expected_status="${3:-0}"
    
    log "${BLUE}Checking $name...${NC}"
    
    if eval "$command" > /dev/null 2>&1; then
        log "${GREEN}✓ $name: OK${NC}"
        return 0
    else
        log "${RED}✗ $name: FAILED${NC}"
        ERRORS+=("$name")
        HEALTH_STATUS=1
        return 1
    fi
}

# Performance check function
check_performance() {
    local name="$1"
    local value="$2"
    local threshold="$3"
    local unit="$4"
    
    if (( $(echo "$value <= $threshold" | bc -l) )); then
        log "${GREEN}✓ $name: $value$unit (OK)${NC}"
        return 0
    else
        log "${YELLOW}⚠ $name: $value$unit (WARNING - threshold: $threshold$unit)${NC}"
        return 1
    fi
}

# Main health check function
main() {
    log "${BLUE}Starting health check...${NC}"
    
    # Check if Docker is running
    check "Docker" "docker info > /dev/null 2>&1"
    
    # Check if sentiment-analysis container is running
    check "Sentiment Analysis Container" "docker ps | grep -q sentiment-analysis"
    
    # Check API health endpoint
    check "API Health Endpoint" "curl -f -s -o /dev/null -w '%{http_code}' $HEALTH_URL | grep -q '200'"
    
    # Check metrics endpoint
    check "Metrics Endpoint" "curl -f -s -o /dev/null -w '%{http_code}' $METRICS_URL | grep -q '200'"
    
    # Check MCP server
    check "MCP Server" "curl -f -s -o /dev/null -w '%{http_code}' $MCP_URL | grep -q '200'"
    
    # Check Advanced Analytics API
    check "Advanced Analytics API" "curl -f -s -o /dev/null -w '%{http_code}' $ADVANCED_ANALYTICS_URL | grep -q '200'"
    
    # Check Main API
    check "Main API" "curl -f -s -o /dev/null -w '%{http_code}' $MAIN_API_URL | grep -q '200'"
    
    # Check Ollama service
    check "Ollama Service" "curl -f -s -o /dev/null -w '%{http_code}' $OLLAMA_URL | grep -q '200'"
    
    # Check Redis service
    check "Redis Service" "docker exec redis redis-cli ping | grep -q 'PONG'"
    
    # Check response time
    log "${BLUE}Checking response time...${NC}"
    RESPONSE_TIME=$(curl -s -o /dev/null -w '%{time_total}' $HEALTH_URL)
    check_performance "API Response Time" "$RESPONSE_TIME" "$MAX_RESPONSE_TIME" "s"
    
    # Check container resource usage
    log "${BLUE}Checking container resource usage...${NC}"
    
    # Memory usage
    MEMORY_USAGE=$(docker stats --no-stream --format "table {{.MemPerc}}" sentiment-analysis | tail -n +2 | sed 's/%//')
    check_performance "Memory Usage" "$MEMORY_USAGE" "$MAX_MEMORY_USAGE" "%"
    
    # CPU usage
    CPU_USAGE=$(docker stats --no-stream --format "table {{.CPUPerc}}" sentiment-analysis | tail -n +2 | sed 's/%//')
    check_performance "CPU Usage" "$CPU_USAGE" "$MAX_CPU_USAGE" "%"
    
    # Check disk usage
    log "${BLUE}Checking disk usage...${NC}"
    DISK_USAGE=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
    check_performance "Disk Usage" "$DISK_USAGE" "$MAX_DISK_USAGE" "%"
    
    # Check log files for errors
    log "${BLUE}Checking log files for errors...${NC}"
    if [ -f "logs/sentiment.log" ]; then
        ERROR_COUNT=$(tail -100 logs/sentiment.log | grep -i "error\|exception\|traceback" | wc -l)
        if [ "$ERROR_COUNT" -eq 0 ]; then
            log "${GREEN}✓ Log Files: No recent errors${NC}"
        else
            log "${YELLOW}⚠ Log Files: $ERROR_COUNT recent errors found${NC}"
            ERRORS+=("Log Errors")
            HEALTH_STATUS=1
        fi
    else
        log "${YELLOW}⚠ Log Files: No log file found${NC}"
    fi
    
    # Check ChromaDB health
    log "${BLUE}Checking ChromaDB health...${NC}"
    if [ -d "cache/chroma_db" ]; then
        CHROMA_SIZE=$(du -sh cache/chroma_db | cut -f1)
        log "${GREEN}✓ ChromaDB: $CHROMA_SIZE${NC}"
    else
        log "${YELLOW}⚠ ChromaDB: Directory not found${NC}"
    fi
    
    # Check model availability
    log "${BLUE}Checking model availability...${NC}"
    MODEL_COUNT=$(curl -s $OLLAMA_URL | jq '.models | length' 2>/dev/null || echo "0")
    if [ "$MODEL_COUNT" -gt 0 ]; then
        log "${GREEN}✓ Models: $MODEL_COUNT models available${NC}"
    else
        log "${RED}✗ Models: No models found${NC}"
        ERRORS+=("No Models")
        HEALTH_STATUS=1
    fi
    
    # Check network connectivity
    log "${BLUE}Checking network connectivity...${NC}"
    check "Internet Connectivity" "ping -c 1 8.8.8.8 > /dev/null 2>&1"
    
    # Check SSL certificate (if using HTTPS)
    if [ -f "nginx/ssl/cert.pem" ]; then
        log "${BLUE}Checking SSL certificate...${NC}"
        CERT_EXPIRY=$(openssl x509 -enddate -noout -in nginx/ssl/cert.pem | cut -d= -f2)
        log "${GREEN}✓ SSL Certificate expires: $CERT_EXPIRY${NC}"
    fi
    
    # Generate health report
    log "${BLUE}Generating health report...${NC}"
    
    if [ $HEALTH_STATUS -eq 0 ]; then
        log "${GREEN}🎉 All health checks passed!${NC}"
        echo "HEALTHY" > /tmp/sentiment-health-status
    else
        log "${RED}❌ Health check failed with the following issues:${NC}"
        for error in "${ERRORS[@]}"; do
            log "${RED}  - $error${NC}"
        done
        echo "UNHEALTHY" > /tmp/sentiment-health-status
    fi
    
    # Create detailed report
    cat > /tmp/sentiment-health-report.txt << EOF
Sentiment Analysis System Health Report
=======================================
Date: $(date)
Status: $(if [ $HEALTH_STATUS -eq 0 ]; then echo "HEALTHY"; else echo "UNHEALTHY"; fi)

System Information:
- Hostname: $(hostname)
- Uptime: $(uptime)
- Load Average: $(uptime | awk -F'load average:' '{print $2}')
- Memory Usage: $MEMORY_USAGE%
- CPU Usage: $CPU_USAGE%
- Disk Usage: $DISK_USAGE%

Service Status:
- Docker: $(if docker info > /dev/null 2>&1; then echo "RUNNING"; else echo "STOPPED"; fi)
- Sentiment Analysis: $(if docker ps | grep -q sentiment-analysis; then echo "RUNNING"; else echo "STOPPED"; fi)
- API Health: $(if curl -f -s -o /dev/null -w '%{http_code}' $HEALTH_URL | grep -q '200'; then echo "OK"; else echo "FAILED"; fi)
- MCP Server: $(if curl -f -s -o /dev/null -w '%{http_code}' $MCP_URL | grep -q '200'; then echo "OK"; else echo "FAILED"; fi)
- Ollama: $(if curl -f -s -o /dev/null -w '%{http_code}' $OLLAMA_URL | grep -q '200'; then echo "OK"; else echo "FAILED"; fi)
- Redis: $(if docker exec redis redis-cli ping | grep -q 'PONG'; then echo "OK"; else echo "FAILED"; fi)

Performance Metrics:
- API Response Time: ${RESPONSE_TIME}s
- Available Models: $MODEL_COUNT
- ChromaDB Size: $CHROMA_SIZE

Recent Errors: $ERROR_COUNT

$(if [ ${#ERRORS[@]} -gt 0 ]; then
    echo "Issues Found:"
    for error in "${ERRORS[@]}"; do
        echo "  - $error"
    done
else
    echo "No issues found"
fi)

EOF
    
    log "${BLUE}Health report saved to: /tmp/sentiment-health-report.txt${NC}"
    
    # Exit with appropriate status code
    exit $HEALTH_STATUS
}

# Handle script arguments
case "${1:-}" in
    --json)
        # JSON output for monitoring systems
        if [ $HEALTH_STATUS -eq 0 ]; then
            echo '{"status":"healthy","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'
        else
            echo '{"status":"unhealthy","errors":['$(printf '"%s"' "${ERRORS[@]}" | paste -sd ',')'],"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'
        fi
        ;;
    --quiet)
        # Quiet mode - only output status
        main > /dev/null 2>&1
        ;;
    --help)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --json    Output JSON format for monitoring"
        echo "  --quiet   Quiet mode - minimal output"
        echo "  --help    Show this help message"
        exit 0
        ;;
    *)
        # Normal mode
        main
        ;;
esac
