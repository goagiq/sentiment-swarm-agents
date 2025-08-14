#!/usr/bin/env python3
"""
Production Deployment Script for Sentiment Analysis Swarm

This script handles the complete production deployment including:
- Environment validation
- Security checks
- Docker image building
- Kubernetes deployment
- Health monitoring
- SSL certificate management
"""

import os
import sys
import subprocess
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """Production deployment orchestrator."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.k8s_dir = self.project_root / "k8s"
        self.namespace = "sentiment-analysis"
        self.deployment_name = "sentiment-analysis"
        
    def validate_environment(self) -> bool:
        """Validate production environment configuration."""
        logger.info("🔍 Validating production environment...")
        
        # Check required environment variables
        required_vars = [
            "NODE_ENV",
            "API_HOST",
            "API_PORT",
            "OLLAMA_HOST",
            "REDIS_HOST"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            return False
        
        # Check if production environment file exists
        env_file = self.project_root / "env.production"
        if not env_file.exists():
            logger.error("❌ Production environment file not found")
            return False
        
        # Validate Kubernetes configuration
        if not self._validate_k8s_config():
            return False
        
        logger.info("✅ Environment validation passed")
        return True
    
    def _validate_k8s_config(self) -> bool:
        """Validate Kubernetes configuration files."""
        logger.info("🔍 Validating Kubernetes configuration...")
        
        required_files = [
            "namespace.yaml",
            "configmap.yaml",
            "secret.yaml",
            "deployment.yaml",
            "service.yaml",
            "ingress.yaml",
            "persistent-volume.yaml",
            "horizontal-pod-autoscaler.yaml"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.k8s_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"❌ Missing Kubernetes files: {missing_files}")
            return False
        
        logger.info("✅ Kubernetes configuration validation passed")
        return True
    
    def build_docker_image(self) -> bool:
        """Build production Docker image."""
        logger.info("🐳 Building production Docker image...")
        
        try:
            # Build with production tag
            cmd = [
                "docker", "build",
                "-t", "sentiment-analysis:latest",
                "-t", "sentiment-analysis:production",
                "--target", "production",
                "."
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error(f"❌ Docker build failed: {result.stderr}")
                return False
            
            logger.info("✅ Docker image built successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker build error: {e}")
            return False
    
    def deploy_to_kubernetes(self) -> bool:
        """Deploy to Kubernetes cluster."""
        logger.info("☸️ Deploying to Kubernetes...")
        
        try:
            # Apply namespace first
            self._apply_k8s_file("namespace.yaml")
            
            # Apply persistent volumes
            self._apply_k8s_file("persistent-volume.yaml")
            
            # Apply secrets and configmaps
            self._apply_k8s_file("secret.yaml")
            self._apply_k8s_file("configmap.yaml")
            
            # Apply deployment
            self._apply_k8s_file("deployment.yaml")
            
            # Apply service
            self._apply_k8s_file("service.yaml")
            
            # Apply ingress
            self._apply_k8s_file("ingress.yaml")
            
            # Apply horizontal pod autoscaler
            self._apply_k8s_file("horizontal-pod-autoscaler.yaml")
            
            logger.info("✅ Kubernetes deployment completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Kubernetes deployment failed: {e}")
            return False
    
    def _apply_k8s_file(self, filename: str) -> bool:
        """Apply a single Kubernetes file."""
        try:
            file_path = self.k8s_dir / filename
            cmd = ["kubectl", "apply", "-f", str(file_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Failed to apply {filename}: {result.stderr}")
                return False
            
            logger.info(f"✅ Applied {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error applying {filename}: {e}")
            return False
    
    def wait_for_deployment(self, timeout: int = 600) -> bool:
        """Wait for deployment to be ready."""
        logger.info("⏳ Waiting for deployment to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                cmd = [
                    "kubectl", "get", "deployment", self.deployment_name,
                    "-n", self.namespace,
                    "-o", "jsonpath={.status.readyReplicas}"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    ready_replicas = int(result.stdout.strip())
                    if ready_replicas >= 3:  # Minimum replicas
                        logger.info("✅ Deployment is ready")
                        return True
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ Error checking deployment status: {e}")
                time.sleep(10)
        
        logger.error("❌ Deployment timeout")
        return False
    
    def run_health_checks(self) -> bool:
        """Run comprehensive health checks."""
        logger.info("🏥 Running health checks...")
        
        # Get service IP
        try:
            cmd = [
                "kubectl", "get", "service", "sentiment-analysis-service",
                "-n", self.namespace,
                "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ Could not get service IP")
                return False
            
            service_ip = result.stdout.strip()
            if not service_ip:
                logger.error("❌ Service IP not available")
                return False
            
            # Test endpoints
            endpoints = [
                f"http://{service_ip}:8003/health",
                f"http://{service_ip}:8003/docs",
                f"http://{service_ip}:8501",
                f"http://{service_ip}:8502"
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, timeout=30)
                    if response.status_code == 200:
                        logger.info(f"✅ Health check passed: {endpoint}")
                    else:
                        logger.warning(f"⚠️ Health check warning: {endpoint} - {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Health check failed: {endpoint} - {e}")
                    return False
            
            logger.info("✅ All health checks passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Setup monitoring and alerting."""
        logger.info("📊 Setting up monitoring...")
        
        try:
            # Deploy Prometheus
            self._apply_k8s_file("monitoring/prometheus.yaml")
            
            # Deploy Grafana
            self._apply_k8s_file("monitoring/grafana.yaml")
            
            # Deploy Jaeger for distributed tracing
            self._apply_k8s_file("monitoring/jaeger.yaml")
            
            logger.info("✅ Monitoring setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {e}")
            return False
    
    def setup_ssl_certificates(self) -> bool:
        """Setup SSL certificates using cert-manager."""
        logger.info("🔒 Setting up SSL certificates...")
        
        try:
            # Check if cert-manager is installed
            cmd = ["kubectl", "get", "namespace", "cert-manager"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning("⚠️ cert-manager not found, installing...")
                self._install_cert_manager()
            
            # Create ClusterIssuer for Let's Encrypt
            self._apply_k8s_file("ssl/cluster-issuer.yaml")
            
            logger.info("✅ SSL certificates setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ SSL setup failed: {e}")
            return False
    
    def _install_cert_manager(self) -> bool:
        """Install cert-manager."""
        try:
            cmd = [
                "kubectl", "apply", "-f",
                "https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ cert-manager installation failed: {result.stderr}")
                return False
            
            logger.info("✅ cert-manager installed")
            return True
            
        except Exception as e:
            logger.error(f"❌ cert-manager installation error: {e}")
            return False
    
    def run_security_scan(self) -> bool:
        """Run security scan on the deployment."""
        logger.info("🔒 Running security scan...")
        
        try:
            # Run Trivy vulnerability scanner
            cmd = [
                "trivy", "image", "--severity", "HIGH,CRITICAL",
                "sentiment-analysis:production"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Security scan failed: {result.stderr}")
                return False
            
            # Check for high/critical vulnerabilities
            if "HIGH" in result.stdout or "CRITICAL" in result.stdout:
                logger.warning("⚠️ Security vulnerabilities found")
                logger.warning(result.stdout)
                return False
            
            logger.info("✅ Security scan passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Security scan error: {e}")
            return False
    
    def deploy(self) -> bool:
        """Execute complete production deployment."""
        logger.info("🚀 Starting production deployment...")
        
        steps = [
            ("Environment Validation", self.validate_environment),
            ("Security Scan", self.run_security_scan),
            ("Docker Build", self.build_docker_image),
            ("Kubernetes Deployment", self.deploy_to_kubernetes),
            ("SSL Setup", self.setup_ssl_certificates),
            ("Monitoring Setup", self.setup_monitoring),
            ("Deployment Wait", self.wait_for_deployment),
            ("Health Checks", self.run_health_checks)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 Executing: {step_name}")
            
            if not step_func():
                logger.error(f"❌ Deployment failed at: {step_name}")
                return False
            
            logger.info(f"✅ Completed: {step_name}")
        
        logger.info("🎉 Production deployment completed successfully!")
        return True


def main():
    """Main deployment function."""
    deployer = ProductionDeployer()
    
    if deployer.deploy():
        logger.info("✅ Production deployment successful")
        sys.exit(0)
    else:
        logger.error("❌ Production deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
