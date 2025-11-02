#!/bin/bash

# Brand Fluidity Analyzer - Backend Setup Script

echo "🚀 Setting up Brand Fluidity Analyzer Backend..."
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install requirements
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python3 << EOF
import nltk
print("Downloading VADER lexicon...")
nltk.download('vader_lexicon', quiet=True)
print("Downloading punkt tokenizer...")
nltk.download('punkt', quiet=True)
print("NLTK data downloaded successfully!")
EOF

# Setup .env file
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ".env file created. Please edit it and add your ANTHROPIC_API_KEY"
else
    echo ".env file already exists"
fi

# Create upload directories
echo ""
echo "Creating upload directories..."
mkdir -p uploads/texts
mkdir -p uploads/images

echo ""
echo "✅ Backend setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your ANTHROPIC_API_KEY"
echo "   Get your key from: https://console.anthropic.com/"
echo ""
echo "2. Run the backend:"
echo "   python main.py"
echo ""
echo "3. Backend will be available at: http://localhost:8000"
echo ""
echo "Happy analyzing! 🎨📊"
