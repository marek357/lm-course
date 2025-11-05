#!/bin/bash

# Colab SSH Setup Script
# This script installs cloudflared and configures SSH for Google Colab access

set -e # Exit on any error

echo "🚀 Setting up Colab SSH access..."
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
Linux*) MACHINE=Linux ;;
Darwin*) MACHINE=Mac ;;
CYGWIN* | MINGW* | MSYS*) MACHINE=Windows ;;
*) MACHINE="UNKNOWN:${OS}" ;;
esac

echo "Detected OS: $MACHINE"
echo ""

# Install cloudflared
if command -v cloudflared &>/dev/null; then
  echo "✓ cloudflared already installed at: $(which cloudflared)"
else
  echo "📦 Installing cloudflared..."

  if [ "$MACHINE" = "Mac" ]; then
    if command -v brew &>/dev/null; then
      brew install cloudflare/cloudflare/cloudflared
    else
      echo "❌ Error: Homebrew not found. Please install Homebrew first:"
      echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
      exit 1
    fi
  elif [ "$MACHINE" = "Linux" ]; then
    # Detect architecture
    ARCH="$(uname -m)"
    if [ "$ARCH" = "x86_64" ]; then
      wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
      sudo dpkg -i cloudflared-linux-amd64.deb
      rm cloudflared-linux-amd64.deb
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
      wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
      sudo dpkg -i cloudflared-linux-arm64.deb
      rm cloudflared-linux-arm64.deb
    else
      echo "❌ Unsupported architecture: $ARCH"
      exit 1
    fi
  elif [ "$MACHINE" = "Windows" ]; then
    echo "❌ Windows detected. Please manually download cloudflared from:"
    echo "   https://github.com/cloudflare/cloudflared/releases"
    exit 1
  else
    echo "❌ Unsupported OS: $MACHINE"
    exit 1
  fi

  echo "✓ cloudflared installed successfully"
fi

# Get cloudflared path
CLOUDFLARED_PATH=$(which cloudflared)
echo ""
echo "cloudflared path: $CLOUDFLARED_PATH"
echo ""

# Setup SSH config
SSH_CONFIG="$HOME/.ssh/config"
mkdir -p "$HOME/.ssh"
touch "$SSH_CONFIG"

# Check if config already exists
if grep -q "Host \*.trycloudflare.com" "$SSH_CONFIG"; then
  echo "⚠️  SSH config for trycloudflare.com already exists in $SSH_CONFIG"
  read -p "Do you want to overwrite it? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping SSH config update"
  else
    # Remove old config
    sed -i.bak '/Host \*.trycloudflare.com/,/ProxyCommand.*cloudflared/d' "$SSH_CONFIG"
    echo "Old config removed"
  fi
fi

# Add new config if not present
if ! grep -q "Host \*.trycloudflare.com" "$SSH_CONFIG"; then
  echo "" >>"$SSH_CONFIG"
  echo "# Cloudflare Tunnel for Colab SSH" >>"$SSH_CONFIG"
  echo "Host *.trycloudflare.com" >>"$SSH_CONFIG"
  echo "    HostName %h" >>"$SSH_CONFIG"
  echo "    User root" >>"$SSH_CONFIG"
  echo "    Port 22" >>"$SSH_CONFIG"
  echo "    ProxyCommand $CLOUDFLARED_PATH access ssh --hostname %h" >>"$SSH_CONFIG"

  echo "✓ SSH config updated at $SSH_CONFIG"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Run the Python code in your Colab notebook to start the SSH server"
echo "2. Copy the hostname from Colab (e.g., something.trycloudflare.com)"
echo "3. Connect using: ssh <hostname>.trycloudflare.com"
echo ""
echo "   Example: ssh johnson-hong-pdt-sets.trycloudflare.com"
echo ""
echo "🎯 For VS Code: Use Remote-SSH extension and paste the hostname"
echo ""
