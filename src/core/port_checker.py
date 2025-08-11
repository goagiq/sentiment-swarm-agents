"""
Port availability checker utility.
"""

import socket
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_port_available(host: str, port: int) -> bool:
    """
    Check if a port is available for binding.
    
    Args:
        host: Host address to check
        port: Port number to check
        
    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
    except Exception as e:
        logger.warning(f"Error checking port {host}:{port}: {e}")
        return False


def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> Optional[int]:
    """
    Find an available port starting from start_port.
    
    Args:
        host: Host address to check
        start_port: Starting port number
        max_attempts: Maximum number of ports to try
        
    Returns:
        Available port number or None if no port found
    """
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(host, port):
            logger.info(f"Found available port: {host}:{port}")
            return port
    
    logger.error(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")
    return None


def get_safe_port(host: str, preferred_port: int) -> int:
    """
    Get a safe port to use, falling back to other ports if needed.
    
    Args:
        host: Host address
        preferred_port: Preferred port number
        
    Returns:
        Available port number
    """
    if is_port_available(host, preferred_port):
        return preferred_port
    
    logger.warning(f"Preferred port {preferred_port} is not available, searching for alternative...")
    
    # Try common alternative ports
    alternative_ports = [8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010]
    
    for port in alternative_ports:
        if is_port_available(host, port):
            logger.info(f"Using alternative port: {port}")
            return port
    
    # If no common ports available, find any available port
    available_port = find_available_port(host, 8000, 100)
    if available_port:
        return available_port
    
    # Last resort - return the original port and let the application handle the error
    logger.error(f"Could not find any available port, using original: {preferred_port}")
    return preferred_port
