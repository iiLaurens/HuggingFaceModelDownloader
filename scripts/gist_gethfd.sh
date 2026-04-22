#!/bin/bash

# HuggingFace Model Downloader - One-liner installer & runner
# Hosted as GitHub Gist, shortened via Cloudflare Worker: https://g.bodaay.io/hfd
#
# Usage:
#   bash <(curl -sSL https://g.bodaay.io/hfd) analyze MODEL -i      # Analyze model (interactive)
#   bash <(curl -sSL https://g.bodaay.io/hfd) download MODEL        # Download a model
#   bash <(curl -sSL https://g.bodaay.io/hfd) serve                 # Start web UI
#   bash <(curl -sSL https://g.bodaay.io/hfd) serve --port 3000     # Start web UI on custom port
#   bash <(curl -sSL https://g.bodaay.io/hfd) install               # Install to ~/.local/bin (no sudo)
#   bash <(curl -sSL https://g.bodaay.io/hfd) install /usr/local/bin  # Install system-wide (may prompt for sudo)
#   bash <(curl -sSL https://g.bodaay.io/hfd) install ~/bin         # Install to any custom path

set -e

# Colors (disabled if NO_COLOR is set or not a terminal)
if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    RED='' GREEN='' YELLOW='' CYAN='' NC=''
fi

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Detect OS and architecture
os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m | tr '[:upper:]' '[:lower:]')

# Normalize architecture names
case "$arch" in
    x86_64)  arch="amd64" ;;
    aarch64) arch="arm64" ;;
    armv7l)  arch="arm" ;;
esac

# GitHub repo and release info
repo="bodaay/HuggingFaceModelDownloader"
binary_name="hfdownloader"

# Check for install command (must be first argument)
install_mode=false
install_path=""
install_path_explicit=false

if [ "$1" = "install" ]; then
    install_mode=true
    shift
    # Optional: custom install path as second argument
    if [ -n "$1" ] && [ "${1:0:1}" != "-" ]; then
        install_path="$1"
        install_path_explicit=true
        shift
    fi
fi

# Pick a sensible default install path when the user didn't specify one.
# Goal: avoid sudo whenever we reasonably can. Prefer a user-owned directory
# that's already in PATH; fall back to ~/.local/bin (and warn about PATH).
# Running as root gets the traditional /usr/local/bin.
pick_default_install_path() {
    if [ "$(id -u 2>/dev/null || echo 0)" = "0" ]; then
        echo "/usr/local/bin"
        return
    fi
    # User directories already in PATH, in priority order.
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) echo "$HOME/.local/bin"; return ;;
        *":$HOME/bin:"*)        echo "$HOME/bin";        return ;;
    esac
    # /usr/local/bin if the user somehow has write access.
    if [ -w "/usr/local/bin" ]; then
        echo "/usr/local/bin"
        return
    fi
    # Final fallback: ~/.local/bin (XDG user-local convention). Not in PATH
    # yet — the script will print instructions.
    echo "$HOME/.local/bin"
}

if [ "$install_mode" = true ] && [ -z "$install_path" ]; then
    install_path="$(pick_default_install_path)"
fi

# All remaining args pass through to hfdownloader
passthrough_args=("$@")

# Fetch latest release tag
info "Fetching latest release..."
latest_tag=$(curl --silent --fail "https://api.github.com/repos/$repo/releases/latest" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$latest_tag" ]; then
    err "Could not fetch latest release tag from GitHub"
    exit 1
fi

info "Latest version: $latest_tag"

# Build download URL
url="https://github.com/${repo}/releases/download/${latest_tag}/${binary_name}_${os}_${arch}_${latest_tag}"
temp_binary="/tmp/${binary_name}_$$"

# Download binary
info "Downloading ${binary_name} for ${os}/${arch}..."
if ! curl -fSL -o "$temp_binary" "$url" 2>/dev/null; then
    err "Download failed from: $url"
    err "Check if binary exists for your platform: ${os}/${arch}"
    rm -f "$temp_binary"
    exit 1
fi
chmod +x "$temp_binary"
ok "Downloaded successfully"

# Install mode: copy to chosen bin directory
if [ "$install_mode" = true ]; then
    info "Installing to ${install_path}..."

    # Create directory if it doesn't exist. sudo is only used when the
    # target is a root-owned path (e.g. user explicitly requested
    # /usr/local/bin) — the default path is a user-owned ~/.local/bin, so
    # no sudo is required in the common case.
    if [ ! -d "$install_path" ]; then
        if ! mkdir -p "$install_path" 2>/dev/null; then
            info "Requesting sudo to create $install_path..."
            sudo mkdir -p "$install_path"
        fi
    fi

    # Move binary to install path
    target="${install_path}/${binary_name}"
    if ! mv "$temp_binary" "$target" 2>/dev/null; then
        info "Requesting sudo to install to $install_path..."
        sudo mv "$temp_binary" "$target"
        sudo chmod +x "$target"
    fi

    ok "Installed: $target"

    # Check if the resolved install path is on PATH. Use a direct PATH
    # match rather than command -v, which can be fooled by an older
    # ${binary_name} already elsewhere in PATH.
    case ":$PATH:" in
        *":$install_path:"*)
            ok "${binary_name} is on your PATH. Run: ${binary_name} --help"
            ;;
        *)
            warn "${install_path} is not on your PATH."
            echo "    Add this to your shell profile (~/.bashrc, ~/.zshrc):"
            echo "        export PATH=\"${install_path}:\$PATH\""
            echo "    Then reload your shell or run: source ~/.bashrc"
            if [ "$install_path_explicit" = false ]; then
                echo "    (or re-run with an explicit path:"
                echo "        bash <(curl -sSL https://g.bodaay.io/hfd) install /usr/local/bin)"
            fi
            ;;
    esac

    # Show version
    "$target" --version 2>/dev/null || true
    exit 0
fi

# Cleanup function for temp binary
cleanup() {
    rm -f "$temp_binary" 2>/dev/null || true
}
trap cleanup EXIT

# Run mode: execute with passed arguments directly from temp binary
# Use 'install' command to install permanently
if [ ${#passthrough_args[@]} -eq 0 ]; then
    exec "$temp_binary" --help
else
    exec "$temp_binary" "${passthrough_args[@]}"
fi
