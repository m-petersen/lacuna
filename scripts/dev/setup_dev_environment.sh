#!/bin/bash
# Quick setup for development environment

set -e

echo "🚀 Setting up development environment..."

# Create directories
mkdir -p notebooks/dev
mkdir -p notebooks/examples
mkdir -p scripts/dev
mkdir -p data/test

# Create .gitkeep files
touch notebooks/dev/.gitkeep
touch notebooks/examples/.gitkeep

echo "✅ Directory structure created"
echo ""
echo "📝 Next steps:"
echo "   1. Run: jupyter lab"
echo "   2. Open: notebooks/dev/test_core_modules.ipynb"
echo "   3. Run all cells to test your implementations"
echo ""
echo "💡 The notebook uses %autoreload - any changes to src/ldk/"
echo "   will be reflected immediately without restarting the kernel!"
