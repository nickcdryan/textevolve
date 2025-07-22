"""
Docker-based sandbox for secure code execution.

This module provides a Docker-based sandbox environment for executing
LLM-generated code safely with proper isolation and resource limits.
"""

import docker
import tempfile
import shutil
import os
import time
import subprocess
import sys
import platform
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def auto_setup_docker() -> tuple:
    """
    Automatically set up Docker based on the current environment.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    system = platform.system().lower()
    
    # First, check if Docker is already working
    if check_docker_available():
        return True, "Docker is already available and running"
    
    # Try platform-specific setup
    if system == "linux":
        return _setup_docker_linux()
    elif system == "darwin":  # macOS
        return _setup_docker_macos()
    elif system == "windows":
        return _setup_docker_windows()
    else:
        return False, f"Unsupported platform: {system}"


def _setup_docker_linux() -> tuple:
    """Set up Docker Engine on Linux systems."""
    try:
        # Check if we're in a container (common in CI/CD)
        if os.path.exists("/.dockerenv"):
            return False, "Running inside a container - Docker-in-Docker not recommended"
        
        # Check if docker is installed
        result = subprocess.run(['which', 'docker'], capture_output=True, text=True)
        if result.returncode != 0:
            # Try to install Docker Engine using the official script
            try:
                logger.info("Installing Docker Engine using official installation script...")
                # Download and run the official Docker installation script
                subprocess.run(['curl', '-fsSL', 'https://get.docker.com', '-o', 'get-docker.sh'], 
                             check=True, timeout=60)
                subprocess.run(['sudo', 'sh', 'get-docker.sh'], 
                             check=True, timeout=300)  # 5 minutes for installation
                
                # Add current user to docker group (requires logout/login to take effect)
                try:
                    subprocess.run(['sudo', 'usermod', '-aG', 'docker', os.getenv('USER', 'root')], 
                                 check=True, timeout=10)
                except subprocess.CalledProcessError:
                    pass  # Non-critical
                
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return False, "Docker not installed and automatic installation failed. Please install Docker Engine manually."
        
        # Try to start Docker service (works on many Linux distros)
        service_started = False
        
        # Try systemctl first (systemd systems)
        try:
            subprocess.run(['sudo', 'systemctl', 'start', 'docker'], 
                         capture_output=True, check=True, timeout=30)
            subprocess.run(['sudo', 'systemctl', 'enable', 'docker'], 
                         capture_output=True, check=True, timeout=10)
            time.sleep(3)  # Give it a moment to start
            service_started = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        
        # Try alternative service management (SysV systems)
        if not service_started:
            try:
                subprocess.run(['sudo', 'service', 'docker', 'start'], 
                             capture_output=True, check=True, timeout=30)
                time.sleep(3)
                service_started = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
        
        # Check if Docker is now available
        if check_docker_available():
            return True, "Docker Engine started successfully"
        elif service_started:
            return False, "Docker service started but daemon not accessible (may need to add user to docker group and logout/login)"
        else:
            return False, "Docker installed but couldn't start the service automatically"
        
    except Exception as e:
        return False, f"Error setting up Docker on Linux: {e}"


def _setup_docker_macos() -> tuple:
    """Set up Docker on macOS systems, prioritizing engine-only solutions."""
    try:
        # PRIORITY 1: Check if Colima is available (lightweight engine-only)
        result = subprocess.run(['which', 'colima'], capture_output=True, text=True)
        if result.returncode == 0:
            try:
                # Start Colima
                subprocess.run(['colima', 'start'], check=True, timeout=60)
                time.sleep(5)
                
                if check_docker_available():
                    return True, "Colima (Docker Engine) started successfully"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
        
        # PRIORITY 2: Try to install Colima via Homebrew (engine-only)
        brew_result = subprocess.run(['which', 'brew'], capture_output=True, text=True)
        if brew_result.returncode == 0:
            try:
                logger.info("Installing Colima (Docker Engine) via Homebrew...")
                # Install Colima and Docker CLI - no GUI
                subprocess.run(['brew', 'install', 'colima', 'docker'], 
                             check=True, timeout=300)  # 5 minutes for install
                
                # Start Colima
                subprocess.run(['colima', 'start'], check=True, timeout=60)
                time.sleep(5)
                
                if check_docker_available():
                    return True, "Colima (Docker Engine) installed and started successfully"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Colima installation failed: {e}")
        
        # FALLBACK: Check if Docker Desktop is available (GUI version)
        docker_app_path = "/Applications/Docker.app"
        if os.path.exists(docker_app_path):
            try:
                logger.info("Falling back to Docker Desktop...")
                subprocess.run(['open', '-a', 'Docker'], check=True, timeout=10)
                
                # Wait for Docker to start (can take a while)
                for i in range(60):  # Wait up to 60 seconds
                    time.sleep(1)
                    if check_docker_available():
                        return True, "Docker Desktop started successfully (consider using Colima for engine-only setup)"
                
                return False, "Docker Desktop launched but didn't become available in time"
            except subprocess.CalledProcessError:
                pass
        
        return False, "Could not set up Docker automatically. Install options: 'brew install colima docker' (recommended) or Docker Desktop."
        
    except Exception as e:
        return False, f"Error setting up Docker on macOS: {e}"


def _setup_docker_windows() -> tuple:
    """Set up Docker on Windows systems."""
    try:
        # Check if Docker Desktop is installed
        docker_paths = [
            "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe",
            "C:\\Program Files (x86)\\Docker\\Docker\\Docker Desktop.exe"
        ]
        
        for docker_path in docker_paths:
            if os.path.exists(docker_path):
                try:
                    subprocess.run([docker_path], timeout=10)
                    
                    # Wait for Docker to start
                    for i in range(30):
                        time.sleep(1)
                        if check_docker_available():
                            return True, "Docker Desktop started successfully"
                    
                    return False, "Docker Desktop launched but didn't become available in time"
                except subprocess.TimeoutExpired:
                    pass
        
        return False, "Docker Desktop not found. Please install Docker Desktop for Windows."
        
    except Exception as e:
        return False, f"Error setting up Docker on Windows: {e}"


class DockerSandbox:
    """Docker-based sandbox for secure code execution."""
    
    def __init__(self, 
                 image: str = "python:3.11-slim",
                 memory_limit: str = "512m",
                 cpu_limit: float = 1.0,
                 timeout: int = 90,
                 network_disabled: bool = True,
                 auto_setup: bool = True):
        """
        Initialize Docker sandbox.
        
        Args:
            image: Docker image to use for execution
            memory_limit: Memory limit (e.g., "512m", "1g")
            cpu_limit: CPU limit as fraction of available cores
            timeout: Execution timeout in seconds
            network_disabled: Whether to disable network access
            auto_setup: Whether to automatically set up Docker if needed
        """
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout
        self.network_disabled = network_disabled
        
        # Try to set up Docker automatically if requested
        if auto_setup and not check_docker_available():
            success, message = auto_setup_docker()
            if success:
                logger.info(f"Docker auto-setup successful: {message}")
            else:
                logger.warning(f"Docker auto-setup failed: {message}")
        
        # Initialize Docker client
        try:
            self.client = docker.from_env()
            # Test Docker connection
            self.client.ping()
        except Exception as e:
            raise RuntimeError(f"Docker not available or not running: {e}")
    
    def check_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        try:
            self.client.ping()
            return True
        except Exception:
            return False
    
    def execute_script(self, script: str, sample: Dict) -> Dict:
        """
        Execute a Python script in a Docker sandbox.
        
        Args:
            script: The Python script content to execute
            sample: The test sample data to pass to the script
            
        Returns:
            Dict with execution results (success, answer/error, output, etc.)
        """
        temp_dir = None
        container = None
        
        try:
            # Create temporary directory for this execution
            temp_dir = tempfile.mkdtemp(prefix="sandbox_")
            temp_path = Path(temp_dir)
            
            # Write script to temp file
            script_path = temp_path / "script.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script)
            
            # Write sample data to temp file
            sample_path = temp_path / "sample.py"
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write(f"sample = {repr(sample)}\n")
            
            # Prepare Docker run configuration
            volumes = {
                str(temp_path): {
                    'bind': '/workspace',
                    'mode': 'rw'
                }
            }
            
            # Set up environment and resource limits
            environment = {}
            
            # Network configuration
            network_mode = 'none' if self.network_disabled else 'bridge'
            
            # Create and start container
            container = self.client.containers.run(
                image=self.image,
                command=[
                    'python', '/workspace/script.py'
                ],
                volumes=volumes,
                working_dir='/workspace',
                mem_limit=self.memory_limit,
                cpu_quota=int(self.cpu_limit * 100000),  # CPU quota in microseconds
                cpu_period=100000,  # 100ms period
                network_mode=network_mode,
                environment=environment,
                detach=True,
                remove=False,  # We'll remove manually after getting logs
                stderr=True,
                stdout=True
            )
            
            # Wait for completion with timeout
            try:
                exit_code = container.wait(timeout=self.timeout)
                
                # Get output
                logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                
                # Parse output similar to original execute_script
                if "ANSWER_START" in logs and "ANSWER_END" in logs:
                    answer = logs.split("ANSWER_START")[1].split("ANSWER_END")[0].strip()
                    return {
                        "success": True,
                        "answer": answer,
                        "output": logs,
                        "sandbox": "docker"
                    }
                elif "ERROR_START" in logs and "ERROR_END" in logs:
                    error = logs.split("ERROR_START")[1].split("ERROR_END")[0].strip()
                    return {
                        "success": False,
                        "error": error,
                        "output": logs,
                        "sandbox": "docker"
                    }
                else:
                    # Check exit code
                    exit_info = exit_code['StatusCode'] if isinstance(exit_code, dict) else exit_code
                    if exit_info == 0:
                        # Script ran but didn't produce expected output format
                        return {
                            "success": False,
                            "error": "Script completed but produced unexpected output format",
                            "output": logs,
                            "sandbox": "docker"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Script exited with code {exit_info}",
                            "output": logs,
                            "sandbox": "docker"
                        }
                        
            except docker.errors.ContainerError as e:
                return {
                    "success": False,
                    "error": f"Container execution failed: {e}",
                    "output": str(e),
                    "sandbox": "docker"
                }
            except Exception as e:
                if "timeout" in str(e).lower():
                    return {
                        "success": False,
                        "error": f"Script execution timed out ({self.timeout} seconds)",
                        "output": "Timeout",
                        "sandbox": "docker"
                    }
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            return {
                "success": False,
                "error": f"Docker execution error: {e}",
                "output": str(e),
                "sandbox": "docker"
            }
            
        finally:
            # Cleanup container
            if container:
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")
            
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory: {e}")
    
    def ensure_image_available(self) -> bool:
        """
        Ensure the Docker image is available, pulling if necessary.
        
        Returns:
            bool: True if image is available, False otherwise
        """
        try:
            # Check if image exists locally
            self.client.images.get(self.image)
            return True
        except docker.errors.ImageNotFound:
            # Try to pull the image
            try:
                logger.info(f"Pulling Docker image: {self.image}")
                self.client.images.pull(self.image)
                return True
            except Exception as e:
                logger.error(f"Failed to pull Docker image {self.image}: {e}")
                return False
        except Exception as e:
            logger.error(f"Error checking Docker image: {e}")
            return False


def check_docker_available() -> bool:
    """
    Check if Docker is available and running.
    
    Returns:
        bool: True if Docker is available, False otherwise
    """
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False