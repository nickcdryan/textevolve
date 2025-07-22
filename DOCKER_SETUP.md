# Docker Engine Setup Guide

This guide shows how to install **Docker Engine** (command-line only, no GUI) for secure code execution in TextEvolve.

## Why Docker Engine (Not Docker Desktop)?

- **Lighter**: No GUI overhead
- **Server-friendly**: Works on remote instances
- **Automated**: Can be scripted and automated
- **Production-ready**: What you'd use in production anyway

---

## Quick Auto-Setup

The system will try to auto-setup Docker for you:

```bash
python verify_setup.py
```

If auto-setup fails, use the manual instructions below.

---

## Manual Installation

### macOS: Colima (Recommended)

```bash
# Install via Homebrew
brew install colima docker

# Start Docker Engine
colima start

# Verify it works
docker run hello-world
```

**Benefits**: Lightweight, no GUI, perfect for development.

### Linux: Official Docker Engine

```bash
# Ubuntu/Debian/CentOS/etc. - Official script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (optional, avoids sudo)
sudo usermod -aG docker $USER

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify it works
docker run hello-world
```

### Windows: Docker Desktop (Simplified)

On Windows, Docker Desktop is the standard approach:

1. Download Docker Desktop from docker.com
2. Install and start it
3. It includes both GUI and engine

### Remote Servers/Cloud

Most cloud providers offer Docker pre-installed or via package managers:

```bash
# AWS Linux
sudo yum install docker
sudo service docker start

# Google Cloud/GCP
sudo apt-get install docker.io
sudo systemctl start docker

# Most other cloud providers
curl -fsSL https://get.docker.com | sh
```

---

## Verification

After installation, verify Docker works:

```bash
# Basic check
docker --version

# Functional check
docker run hello-world

# Check daemon is running
docker info
```

---

## Troubleshooting

### macOS: "docker: command not found"
- Make sure you installed both `colima` AND `docker` CLI
- Try: `brew install docker`

### Linux: "permission denied"
- Add yourself to docker group: `sudo usermod -aG docker $USER`
- Log out and back in, or use `newgrp docker`

### Any Platform: "Cannot connect to Docker daemon"
- Make sure the Docker service is running
- macOS: `colima start`
- Linux: `sudo systemctl start docker`

### Still having issues?
Run the verification script for detailed diagnostics:

```bash
python verify_setup.py
```

---

## Why Not Docker Desktop?

Docker Desktop is fine, but has downsides:

- ❌ **GUI required**: Needs manual clicking/opening
- ❌ **Resource heavy**: Runs a full Linux VM
- ❌ **License restrictions**: Commercial use restrictions
- ❌ **Remote unfriendly**: Doesn't work on headless servers

Docker Engine gives you the same functionality with none of these issues. 