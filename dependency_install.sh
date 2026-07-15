#!/usr/bin/env bash
set -euo pipefail

# ── System packages ──────────────────────────────────────────────
sudo apt-get update
sudo apt-get install -y \
  python3.12-venv python3-pip python3-dev \
  build-essential \
  git curl wget jq unzip \
  sqlite3 libsqlite3-dev \
  htop tmux

# ── Docker (official repo — avoid the snap, it has confinement issues
#    with bind mounts and compose) ────────────────────────────────
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER   # re-login (or `newgrp docker`) after this

# ── Ollama (native install, not containerized — simpler for the labs;
#    you'll containerize it in the capstone compose file) ─────────
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull nomic-embed-text

# ── Python environment ───────────────────────────────────────────
mkdir -p ~/langchain-course && cd ~/langchain-course
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U \
  langchain langchain-anthropic langchain-ollama \
  langchain-community langchain-text-splitters \
  langchain-chroma langchain-qdrant \
  langgraph \
  pypdf unstructured[md] \
  atlassian-python-api \
  python-dotenv ipython
