#!/bin/bash
# media-match install/uninstall/update script

set -e

PACKAGE_NAME="mlx-media-match"
VENV_DIR=".venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_venv() {
    if [ ! -d "${VENV_DIR}" ]; then
        print_error "Virtual environment not found at ${VENV_DIR}"
        echo "Create it first with: python3 -m venv ${VENV_DIR}"
        exit 1
    fi

    if [ ! -f "${PYTHON}" ]; then
        print_error "Python not found in virtual environment"
        exit 1
    fi
}

install() {
    check_venv
    print_status "Installing ${PACKAGE_NAME} in editable mode..."
    ${PIP} install -e .
    print_status "Installation complete!"
    print_status "You can now run: media-match --help"
}

uninstall() {
    check_venv
    print_status "Uninstalling ${PACKAGE_NAME}..."
    ${PIP} uninstall -y ${PACKAGE_NAME}
    print_status "Uninstallation complete!"
}

update() {
    check_venv
    print_status "Updating ${PACKAGE_NAME}..."
    ${PIP} install -e . --upgrade
    print_status "Update complete!"
}

reinstall() {
    check_venv
    print_status "Reinstalling ${PACKAGE_NAME}..."
    ${PIP} uninstall -y ${PACKAGE_NAME} 2>/dev/null || true
    ${PIP} install -e .
    print_status "Reinstallation complete!"
}

clean() {
    check_venv
    print_status "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info .pytest_cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_status "Clean complete!"
}

status() {
    check_venv
    print_status "Checking installation status..."
    ${PIP} show ${PACKAGE_NAME} 2>/dev/null || print_warning "${PACKAGE_NAME} is not installed"
}

# Main command handler
case "${1:-install}" in
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    update)
        update
        ;;
    reinstall)
        reinstall
        ;;
    clean)
        clean
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {install|uninstall|update|reinstall|clean|status}"
        echo ""
        echo "Commands:"
        echo "  install    - Install the package in editable mode"
        echo "  uninstall  - Remove the package"
        echo "  update     - Update/reinstall the package"
        echo "  reinstall  - Uninstall then install fresh"
        echo "  clean      - Remove build artifacts and cache files"
        echo "  status     - Check if package is installed"
        exit 1
        ;;
esac
