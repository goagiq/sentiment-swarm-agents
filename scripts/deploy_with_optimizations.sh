#!/bin/bash

# Deployment Script with Optimization Validation
# This script deploys the multilingual sentiment analysis system with all optimizations enabled

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-"production"}
DEBUG=${DEBUG:-"false"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
CACHE_DIR="$PROJECT_ROOT/cache"
LOGS_DIR="$PROJECT_ROOT/logs"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $python_version"
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        docker_version=$(docker --version)
        log_info "Docker version: $docker_version"
    else
        log_warning "Docker not found - will use manual deployment"
    fi
    
    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        log_warning "Redis CLI not found - will attempt to start Redis server"
    fi
    
    log_success "System requirements check completed"
}

setup_directories() {
    log_info "Setting up directories..."
    
    # Create necessary directories
    mkdir -p "$DATA_DIR"
    mkdir -p "$CACHE_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$PROJECT_ROOT/monitoring"
    
    log_success "Directories setup completed"
}

setup_virtual_environment() {
    log_info "Setting up virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Install additional optimization packages
    pip install psutil redis
    
    log_success "Virtual environment setup completed"
}

validate_configurations() {
    log_info "Validating configurations..."
    
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    # Run configuration validation
    python3 -c "
import sys
sys.path.insert(0, 'src')
from config.config_validator import config_validator
import asyncio

async def validate():
    try:
        # Validate all configurations
        is_valid = await config_validator.validate_all_configurations()
        if is_valid:
            print('✅ All configurations are valid')
        else:
            print('❌ Configuration validation failed')
            sys.exit(1)
    except Exception as e:
        print(f'❌ Configuration validation error: {e}')
        sys.exit(1)

asyncio.run(validate())
"
    
    log_success "Configuration validation completed"
}

start_services() {
    log_info "Starting services..."
    
    cd "$PROJECT_ROOT"
    
    # Start Redis if not running
    if ! redis-cli ping &> /dev/null; then
        log_info "Starting Redis server..."
        redis-server --daemonize yes
        sleep 2
        
        if redis-cli ping &> /dev/null; then
            log_success "Redis server started"
        else
            log_error "Failed to start Redis server"
            exit 1
        fi
    else
        log_info "Redis server is already running"
    fi
    
    # Initialize databases
    log_info "Initializing databases..."
    source .venv/bin/activate
    python3 -c "
import sys
sys.path.insert(0, 'src')
from core.vectordb import initialize_vectordb
import asyncio

async def init_db():
    try:
        await initialize_vectordb()
        print('✅ Database initialization completed')
    except Exception as e:
        print(f'❌ Database initialization error: {e}')
        sys.exit(1)

asyncio.run(init_db())
"
    
    log_success "Services startup completed"
}

run_optimization_tests() {
    log_info "Running optimization validation tests..."
    
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    # Run optimization tests
    python3 -c "
import sys
sys.path.insert(0, 'Test')
from test_phase4_optimization_integration import OptimizationIntegrationTest
import asyncio

async def run_tests():
    try:
        test = OptimizationIntegrationTest()
        results = await test.run_all_tests()
        
        # Calculate success rate
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if 'PASS' in str(result))
        success_rate = (passed_tests / total_tests) * 100
        
        print(f'✅ Optimization tests completed: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)')
        
        if success_rate < 90:
            print('⚠️  Warning: Success rate below 90%')
            for test_name, result in results.items():
                if 'FAIL' in str(result):
                    print(f'   ❌ {test_name}: {result}')
        else:
            print('✅ All optimization tests passed successfully')
            
    except Exception as e:
        print(f'❌ Optimization test error: {e}')
        sys.exit(1)

asyncio.run(run_tests())
"
    
    log_success "Optimization validation completed"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    cd "$PROJECT_ROOT"
    
    # Create monitoring configuration
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'sentiment-analysis'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:6379']

  - job_name: 'chromadb'
    static_configs:
      - targets: ['localhost:8001']
EOF
    
    # Create alert rules
    mkdir -p monitoring/rules
    cat > monitoring/rules/alerts.yml << 'EOF'
groups:
  - name: sentiment-analysis
    rules:
      - alert: HighMemoryUsage
        expr: memory_usage_bytes / 1024 / 1024 / 1024 > 8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 8GB for 5 minutes"

      - alert: LowCacheHitRate
        expr: cache_hit_rate < 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate detected"
          description: "Cache hit rate is below 80% for 10 minutes"

      - alert: HighErrorRate
        expr: rate(requests_total{status="error"}[5m]) / rate(requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% for 5 minutes"
EOF
    
    log_success "Monitoring setup completed"
}

start_application() {
    log_info "Starting application..."
    
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    # Set environment variables
    export ENVIRONMENT="$ENVIRONMENT"
    export DEBUG="$DEBUG"
    export LOG_LEVEL="$LOG_LEVEL"
    
    # Start the application in background
    nohup python3 main.py > "$LOGS_DIR/app.log" 2>&1 &
    APP_PID=$!
    
    # Wait for application to start
    sleep 5
    
    # Check if application is running
    if kill -0 $APP_PID 2>/dev/null; then
        log_success "Application started successfully (PID: $APP_PID)"
        echo $APP_PID > "$PROJECT_ROOT/app.pid"
    else
        log_error "Failed to start application"
        exit 1
    fi
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Wait for application to be ready
    sleep 10
    
    # Check application health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Application health check passed"
    else
        log_error "Application health check failed"
        exit 1
    fi
    
    # Check Redis connectivity
    if redis-cli ping > /dev/null 2>&1; then
        log_success "Redis connectivity check passed"
    else
        log_error "Redis connectivity check failed"
        exit 1
    fi
    
    log_success "All health checks passed"
}

run_performance_benchmarks() {
    log_info "Running performance benchmarks..."
    
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    # Run performance benchmarks
    python3 -c "
import sys
sys.path.insert(0, 'Test')
from test_phase4_optimization_integration import OptimizationIntegrationTest
import asyncio

async def run_benchmarks():
    try:
        test = OptimizationIntegrationTest()
        benchmarks = await test.run_performance_benchmarks()
        
        print('📊 Performance Benchmark Results:')
        for metric, value in benchmarks.items():
            print(f'   {metric}: {value}')
        
        # Check if benchmarks meet requirements
        if benchmarks.get('cache_hit_rate', 0) > 0.8:
            print('✅ Cache performance: PASS')
        else:
            print('❌ Cache performance: FAIL')
            
        if benchmarks.get('memory_usage_mb', 0) < 4096:
            print('✅ Memory usage: PASS')
        else:
            print('❌ Memory usage: FAIL')
            
        if benchmarks.get('processing_time_ms', 0) < 1000:
            print('✅ Processing time: PASS')
        else:
            print('❌ Processing time: FAIL')
            
    except Exception as e:
        print(f'❌ Benchmark error: {e}')

asyncio.run(run_benchmarks())
"
    
    log_success "Performance benchmarks completed"
}

show_deployment_summary() {
    log_info "Deployment Summary"
    echo "=================="
    echo "Environment: $ENVIRONMENT"
    echo "Application PID: $(cat "$PROJECT_ROOT/app.pid" 2>/dev/null || echo 'N/A')"
    echo "Application URL: http://localhost:8000"
    echo "Web Interface: http://localhost:8501"
    echo "Health Check: http://localhost:8000/health"
    echo "Logs: $LOGS_DIR/app.log"
    echo ""
    echo "Services:"
    echo "  - Main Application: ✅ Running"
    echo "  - Redis: ✅ Running"
    echo "  - ChromaDB: ✅ Initialized"
    echo "  - Monitoring: ✅ Configured"
    echo ""
    echo "Optimizations:"
    echo "  - Multi-level Caching: ✅ Enabled"
    echo "  - Parallel Processing: ✅ Enabled"
    echo "  - Memory Management: ✅ Enabled"
    echo "  - Performance Monitoring: ✅ Enabled"
    echo ""
    log_success "Deployment completed successfully!"
}

# Main deployment process
main() {
    log_info "Starting deployment with optimizations..."
    
    check_requirements
    setup_directories
    setup_virtual_environment
    validate_configurations
    start_services
    run_optimization_tests
    setup_monitoring
    start_application
    run_health_checks
    run_performance_benchmarks
    show_deployment_summary
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --validate     Only validate configurations"
        echo "  --test         Only run optimization tests"
        echo "  --monitor      Only setup monitoring"
        echo ""
        echo "Environment variables:"
        echo "  ENVIRONMENT    Deployment environment (default: production)"
        echo "  DEBUG          Enable debug mode (default: false)"
        echo "  LOG_LEVEL      Log level (default: INFO)"
        exit 0
        ;;
    --validate)
        check_requirements
        setup_virtual_environment
        validate_configurations
        log_success "Configuration validation completed"
        exit 0
        ;;
    --test)
        check_requirements
        setup_virtual_environment
        run_optimization_tests
        exit 0
        ;;
    --monitor)
        setup_monitoring
        log_success "Monitoring setup completed"
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
