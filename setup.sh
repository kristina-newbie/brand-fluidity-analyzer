#!/bin/bash

# Brand Fluidity Analyzer - Setup (run from project root)

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "🚀 Setting up Brand Fluidity Analyzer..."
echo ""

echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
# Require Python 3.10–3.12 (gensim and others lack 3.14 wheels / C-API support)
python_major=$(echo "$python_version" | cut -d. -f1)
python_minor=$(echo "$python_version" | cut -d. -f2)
if [ "$python_major" = "3" ] && [ -n "$python_minor" ] && [ "$python_minor" -ge 14 ]; then
    echo ""
    echo "⚠️  Python 3.14+ is not supported yet (gensim and some deps have no compatible builds)."
    echo "   Use Python 3.10, 3.11, or 3.12. Example with pyenv:"
    echo "     pyenv install 3.12.0 && pyenv local 3.12.0"
    echo "   Then re-run ./setup.sh"
    exit 1
fi
if [ "$python_major" = "3" ] && [ -n "$python_minor" ] && [ "$python_minor" -lt 10 ]; then
    echo ""
    echo "⚠️  Python 3.10 or newer is required."
    exit 1
fi
echo ""

read -p "Create virtual environment .venv at project root? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

echo ""
echo "Installing Python dependencies..."
# Use the same interpreter for pip and later steps so NLTK is found
if [ -d .venv ]; then
    PY=.venv/bin/python3
else
    PY=python3
fi
"$PY" -m pip install -r backend/requirements-dev.txt

echo ""
echo "Downloading NLTK data..."
"$PY" -c "
import nltk
print('Downloading VADER lexicon...')
nltk.download('vader_lexicon', quiet=True)
print('Downloading punkt tokenizer...')
nltk.download('punkt', quiet=True)
print('NLTK data downloaded.')
"

echo ""
if [ ! -f backend/.env ]; then
    echo "Creating backend/.env from template..."
    cp backend/.env.example backend/.env
    echo "Edit backend/.env and add your ANTHROPIC_API_KEY"
else
    echo "backend/.env already exists"
fi

echo ""
echo "Creating upload directories..."
mkdir -p backend/uploads/texts
mkdir -p backend/uploads/images

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit backend/.env and add ANTHROPIC_API_KEY"
echo "  2. Activate venv if needed: source .venv/bin/activate"
echo "  3. Run backend: cd backend && python main.py"
echo "  4. Lint: pylint backend"
echo ""
