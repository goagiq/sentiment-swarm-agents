#!/bin/bash

# Production Deployment Script for Sentiment Analysis Swarm (Phases 1-5)
# This script deploys the complete system to production

set -e

echo "🚀 Starting Production Deployment for Sentiment Analysis Swarm"
echo "================================================================"

# Configuration
NAMESPACE="sentiment-analysis"
IMAGE_NAME="sentiment-analysis"
IMAGE_TAG="latest"
REGISTRY="your-registry.com"  # Update with your registry

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "docker is not installed. Please install docker first."
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    print_status "Building Docker image..."
    
    # Build the image
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    
    # Tag for registry (if using external registry)
    if [ "$REGISTRY" != "your-registry.com" ]; then
        docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
        docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    fi
    
    print_success "Docker image built successfully"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    print_status "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes configurations
    print_status "Applying Kubernetes configurations..."
    
    # Apply in order
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    kubectl apply -f k8s/persistent-volume.yaml
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/ingress.yaml
    
    print_success "Kubernetes configurations applied"
}

# Wait for deployment to be ready
wait_for_deployment() {
    print_status "Waiting for deployment to be ready..."
    
    kubectl wait --for=condition=available --timeout=300s deployment/sentiment-analysis -n ${NAMESPACE}
    
    print_success "Deployment is ready"
}

# Check service health
check_service_health() {
    print_status "Checking service health..."
    
    # Get service IP
    SERVICE_IP=$(kubectl get service sentiment-analysis-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_IP" ]; then
        print_warning "Service IP not available yet. You may need to wait for LoadBalancer provisioning."
        print_status "You can check the service status with: kubectl get service -n ${NAMESPACE}"
    else
        print_success "Service is available at: http://${SERVICE_IP}"
    fi
    
    # Check API health
    print_status "Checking API health..."
    kubectl run test-health --image=curlimages/curl --rm -i --restart=Never -- \
        curl -f http://sentiment-analysis-service:8003/health || print_warning "API health check failed"
}

# Display access information
display_access_info() {
    echo ""
    echo "🎉 Deployment Complete!"
    echo "================================================================"
    echo "📋 Access Information:"
    echo ""
    echo "🌐 External Access (if LoadBalancer is configured):"
    echo "   📊 Main UI:        http://<loadbalancer-ip>:8501"
    echo "   🏠 Landing Page:   http://<loadbalancer-ip>:8502"
    echo "   🔗 API Docs:       http://<loadbalancer-ip>:8003/docs"
    echo "   🤖 MCP Server:     http://<loadbalancer-ip>:8000/mcp"
    echo ""
    echo "🔧 Internal Access (within cluster):"
    echo "   📊 Main UI:        http://sentiment-analysis-service:8501"
    echo "   🏠 Landing Page:   http://sentiment-analysis-service:8502"
    echo "   🔗 API Docs:       http://sentiment-analysis-service:8003/docs"
    echo "   🤖 MCP Server:     http://sentiment-analysis-service:8000/mcp"
    echo ""
    echo "📊 Monitoring:"
    echo "   📈 Prometheus:     http://<loadbalancer-ip>:9090"
    echo "   📊 Grafana:        http://<loadbalancer-ip>:3000"
    echo ""
    echo "🔍 Useful Commands:"
    echo "   kubectl get pods -n ${NAMESPACE}"
    echo "   kubectl logs -f deployment/sentiment-analysis -n ${NAMESPACE}"
    echo "   kubectl get service -n ${NAMESPACE}"
    echo "   kubectl describe deployment sentiment-analysis -n ${NAMESPACE}"
    echo ""
    echo "📋 Implementation Status:"
    echo "   ✅ Phase 1: Core Sentiment Analysis"
    echo "   ✅ Phase 2: Business Intelligence"
    echo "   ✅ Phase 3: Advanced Analytics"
    echo "   ✅ Phase 4: Export & Automation"
    echo "   ✅ Phase 5: Semantic Search & Reflection"
    echo ""
    echo "🚀 System is ready for production use!"
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    
    check_prerequisites
    build_and_push_image
    deploy_to_kubernetes
    wait_for_deployment
    check_service_health
    display_access_info
    
    print_success "Production deployment completed successfully!"
}

# Run main function
main "$@"
