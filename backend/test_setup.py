"""
Test script to verify Brand Fluidity Analyzer setup
Run this after installation to check if all components are working.
"""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    packages = {
        'fastapi': 'FastAPI',
        'torch': 'PyTorch',
        'clip': 'CLIP',
        'sentence_transformers': 'Sentence Transformers',
        'sklearn': 'scikit-learn',
        'cv2': 'OpenCV',
        'nltk': 'NLTK',
        'gensim': 'Gensim',
        'PIL': 'Pillow',
        'anthropic': 'Anthropic',
        'scipy': 'SciPy'
    }

    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)

    return len(failed) == 0, failed

def test_nltk_data():
    """Test if NLTK data is downloaded."""
    print("\nTesting NLTK data...")
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        from nltk.tokenize import sent_tokenize

        sia = SentimentIntensityAnalyzer()
        tokens = sent_tokenize("This is a test sentence. This is another one.")
        print(f"  ✓ NLTK data available")
        return True
    except Exception as e:
        print(f"  ✗ NLTK data missing - {e}")
        print("  Run: python -c \"import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')\"")
        return False

def test_models():
    """Test if models can be loaded."""
    print("\nTesting model loading...")

    # Test CLIP
    try:
        import clip
        import torch
        print("  Loading CLIP model (this may take a moment)...")
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        print("  ✓ CLIP model loaded")
    except Exception as e:
        print(f"  ✗ CLIP model failed - {e}")
        return False

    # Test Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading Sentence Transformer model...")
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
        print("  ✓ Sentence Transformer loaded")
    except Exception as e:
        print(f"  ✗ Sentence Transformer failed - {e}")
        return False

    return True

def test_analyzer():
    """Test if BrandFluidityAnalyzerV2 can be instantiated."""
    print("\nTesting BrandFluidityAnalyzerV2...")
    try:
        from brand_fluidity_analyzer_v2 import BrandFluidityAnalyzerV2
        analyzer = BrandFluidityAnalyzerV2()
        print("  ✓ BrandFluidityAnalyzerV2 initialized successfully")
        return True
    except Exception as e:
        print(f"  ✗ BrandFluidityAnalyzerV2 failed - {e}")
        return False

def test_api_key():
    """Check if Anthropic API key is configured."""
    print("\nChecking Anthropic API key...")
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key and api_key != "your_anthropic_api_key_here":
        print("  ✓ ANTHROPIC_API_KEY is configured")
        return True
    else:
        print("  ⚠ ANTHROPIC_API_KEY not configured in .env")
        print("    LLM insights will not be available")
        print("    Get your key from: https://console.anthropic.com/")
        return False

def main():
    print("=" * 60)
    print("Brand Fluidity Analyzer v2.0 - Setup Verification")
    print("=" * 60)
    print()

    all_passed = True

    # Test imports
    imports_ok, failed = test_imports()
    if not imports_ok:
        print(f"\n⚠ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt --break-system-packages")
        all_passed = False

    # Test NLTK
    if not test_nltk_data():
        all_passed = False

    # Test models
    if not test_models():
        all_passed = False

    # Test analyzer
    if not test_analyzer():
        all_passed = False

    # Test API key
    api_key_ok = test_api_key()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All core components working!")
        if not api_key_ok:
            print("⚠  Note: Configure ANTHROPIC_API_KEY for AI insights")
        print("\nYou can now run: python main.py")
    else:
        print("❌ Some components failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
