"""
Sample Data Generator for Brand Fluidity Analyzer
Creates synthetic brand materials for testing the application.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_sample_texts(output_dir="sample_data/texts"):
    """Generate sample brand texts across different years."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    texts = {
        "2020_manifesto.txt": """
        Our brand stands for innovation and sustainability.
        We believe in creating products that make a difference.
        Quality, integrity, and customer satisfaction are our core values.
        Join us in building a better future for everyone.
        """,
        
        "2021_campaign.txt": """
        Innovate. Inspire. Impact.
        We're not just a brand - we're a movement.
        Our commitment to sustainability drives everything we do.
        Together, we're creating positive change in the world.
        Experience the difference quality makes.
        """,
        
        "2022_social_media.txt": """
        Sustainability isn't just a buzzword - it's our DNA. 🌱
        Innovation meets responsibility in everything we create.
        Join our community of changemakers and visionaries.
        Quality you can trust, impact you can see.
        #BrandForGood #SustainableFuture
        """,
        
        "2023_website.txt": """
        Welcome to the future of sustainable innovation.
        We're pioneering new approaches to quality and responsibility.
        Our products reflect our commitment to environmental stewardship.
        Be part of the solution. Choose sustainability. Choose quality.
        Experience innovation that makes a real difference.
        """
    }
    
    for filename, content in texts.items():
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(content.strip())
    
    print(f"✓ Created {len(texts)} sample text files in {output_dir}")
    return list(texts.keys())

def create_sample_images(output_dir="sample_data/images"):
    """Generate simple sample images with varying colors and styles."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Different color schemes for different years
    color_schemes = {
        "2020": [(70, 130, 180), (100, 149, 237)],  # Blue tones
        "2021": [(60, 179, 113), (46, 139, 87)],    # Green tones
        "2022": [(147, 112, 219), (138, 43, 226)],  # Purple tones
        "2023": [(255, 140, 0), (255, 165, 0)]      # Orange tones
    }
    
    image_files = []
    
    for year, colors in color_schemes.items():
        for i in range(3):
            # Create image
            img = Image.new('RGB', (800, 600), color=colors[i % len(colors)])
            draw = ImageDraw.Draw(img)
            
            # Add some shapes for visual diversity
            if i == 0:
                # Circle
                draw.ellipse([200, 150, 600, 450], fill=colors[(i+1) % len(colors)])
            elif i == 1:
                # Rectangle
                draw.rectangle([150, 100, 650, 500], fill=colors[(i+1) % len(colors)])
            else:
                # Triangle
                draw.polygon([(400, 100), (650, 500), (150, 500)], fill=colors[(i+1) % len(colors)])
            
            # Add year text
            try:
                # Try to use a default font, but don't fail if not available
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            draw.text((350, 270), year, fill=(255, 255, 255), font=font)
            
            # Save image
            filename = f"{year}_visual_{i+1}.jpg"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, quality=95)
            image_files.append(filename)
    
    print(f"✓ Created {len(image_files)} sample images in {output_dir}")
    return image_files

def create_readme(output_dir="sample_data"):
    """Create a README explaining the sample data."""
    
    readme_content = """# Sample Brand Data

This directory contains synthetic brand materials for testing the Brand Fluidity Analyzer.

## Contents

### Text Files (texts/)
- **2020_manifesto.txt**: Early brand messaging focused on core values
- **2021_campaign.txt**: Campaign copy with evolved positioning
- **2022_social_media.txt**: Social media content with modern voice
- **2023_website.txt**: Current website copy with refined messaging

### Image Files (images/)
- 12 synthetic images across 4 years (2020-2023)
- Color schemes evolve from blue → green → purple → orange
- Different geometric compositions represent visual evolution

## Usage

1. Go to the Brand Fluidity Analyzer app
2. Enter brand name: "Sample Brand"
3. Upload all text files from texts/ directory
4. Upload all images from images/ directory
5. Click "Analyze Brand Fluidity"

## Expected Results

This sample dataset should show:
- **TFI (Textual Fluidity)**: Moderate (~0.4-0.6)
  - Consistent sustainability theme with evolving language
- **VFI (Visual Fluidity)**: High (~0.6-0.8)
  - Significant color palette changes over time
- **CMCS (Coherence)**: Good (~0.6-0.8)
  - Text and visuals aligned on sustainability theme
- **BFI (Overall)**: Semi-fluid (~0.5-0.7)
  - Adaptive brand identity

## Customization

Feel free to:
- Edit text files to test different messaging patterns
- Add more images with your own brand visuals
- Mix years to test different evolution scenarios

---

Generated by Brand Fluidity Analyzer Sample Data Generator
"""
    
    with open(os.path.join(output_dir, "README.md"), 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Created README in {output_dir}")

def main():
    print("=" * 60)
    print("Brand Fluidity Analyzer - Sample Data Generator")
    print("=" * 60)
    print()
    
    base_dir = "sample_data"
    
    # Generate texts
    text_files = create_sample_texts(os.path.join(base_dir, "texts"))
    
    # Generate images
    image_files = create_sample_images(os.path.join(base_dir, "images"))
    
    # Create README
    create_readme(base_dir)
    
    print()
    print("=" * 60)
    print("✅ Sample data generated successfully!")
    print()
    print(f"Location: ./{base_dir}/")
    print(f"  - {len(text_files)} text files")
    print(f"  - {len(image_files)} image files")
    print()
    print("Next steps:")
    print("1. Start the backend: python main.py")
    print("2. Open the frontend in your browser")
    print("3. Upload the sample files and test the analyzer")
    print("=" * 60)

if __name__ == "__main__":
    main()
