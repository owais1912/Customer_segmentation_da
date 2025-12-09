"""
Create smaller sample files for GitHub deployment
Run this script to create deployment-ready data files
"""
import pandas as pd
from pathlib import Path

outputs = Path("outputs")

print("Creating sample data files for deployment...")
print("=" * 50)

# List of large files to sample
large_files = [
    "master_segments_kmeans.csv",
    "master_segments_enriched.csv", 
    "master_segments_with_anomaly.csv",
    "master_segments.csv",
    "rfm_base.csv",
    "rfm_with_sentiment.csv"
]

for filename in large_files:
    filepath = outputs / filename
    if filepath.exists():
        # Read the file
        df = pd.read_csv(filepath)
        original_size = filepath.stat().st_size / (1024 * 1024)  # MB
        
        # Take first 5000 rows (should be well under 100MB)
        sample_df = df.head(5000)
        
        # Save with _sample suffix
        sample_path = outputs / filename.replace('.csv', '_sample.csv')
        sample_df.to_csv(sample_path, index=False)
        
        sample_size = sample_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"✓ {filename}")
        print(f"  Original: {original_size:.2f} MB ({len(df):,} rows)")
        print(f"  Sample: {sample_size:.2f} MB ({len(sample_df):,} rows)")
        print()

print("=" * 50)
print("✅ Sample files created!")
print("\nFor deployment, upload the *_sample.csv files instead of the originals.")
print("The app will work the same way with the sample data.")
