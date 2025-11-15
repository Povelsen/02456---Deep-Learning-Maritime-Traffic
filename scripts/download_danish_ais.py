#!/usr/bin/env python3
"""
Danish AIS Data Downloader
=========================

Download AIS data from the Danish AIS database for extended time periods.

Note: This script provides guidance on accessing Danish AIS data. 
Actual implementation may require API keys or specific access methods.
"""

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time


def get_danish_ais_urls():
    """
    Get URLs for Danish AIS data downloads.
    
    Returns:
        Dictionary with data source information
    """
    # Danish AIS data sources (these may change, check current availability)
    sources = {
        "dma_primary": {
            "name": "Danish Maritime Authority",
            "base_url": "https://web.ais.dk/aisdata/",
            "description": "Official Danish AIS data portal",
            "requires_registration": True
        },
        "alternative_sources": [
            {
                "name": "MarineTraffic Historical Data",
                "url": "https://www.marinetraffic.com/en/data/historical-data",
                "note": "Commercial service, may require subscription"
            },
            {
                "name": "AISHub",
                "url": "https://www.aishub.net/",
                "note": "AIS data aggregator"
            }
        ]
    }
    
    return sources


def generate_sample_data_for_week(start_date: str, output_dir: str):
    """
    Generate sample AIS data for a week (for demonstration/testing).
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        output_dir: Directory to save generated files
        
    Returns:
        List of generated CSV file paths
    """
    print(f"📅 Generating sample AIS data for week starting {start_date}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    generated_files = []
    
    for i in range(7):
        current_date = start_dt + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Generate sample data for this day
        sample_data = generate_daily_ais_sample(date_str)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"aisdk_{date_str}.csv")
        sample_data.to_csv(output_file, index=False)
        generated_files.append(output_file)
        
        print(f"   Generated: {output_file}")
    
    return generated_files


def generate_daily_ais_sample(date_str: str) -> pd.DataFrame:
    """
    Generate sample AIS data for a single day.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        
    Returns:
        DataFrame with sample AIS data
    """
    np.random.seed(hash(date_str) % 2**32)
    
    # Generate sample vessel data
    n_vessels = np.random.randint(50, 150)
    n_records = np.random.randint(5000, 15000)
    
    # Vessel MMSIs (Danish maritime identification digits)
    danish_mids = [2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 219, 220]
    mmsis = []
    for _ in range(n_vessels):
        mid = str(np.random.choice(danish_mids))
        mmsi = mid + ''.join([str(np.random.randint(0, 9)) for _ in range(9-len(mid))])
        mmsis.append(mmsi)
    
    # Generate timestamps throughout the day
    base_date = datetime.strptime(date_str, "%Y-%m-%d")
    timestamps = []
    for _ in range(n_records):
        random_time = base_date + timedelta(
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        timestamps.append(random_time.strftime("%d/%m/%Y %H:%M:%S"))
    
    # Generate vessel positions (around Denmark)
    # Denmark approximate bounds: 54.5°N to 58°N, 7.5°E to 15.5°E
    latitudes = np.random.uniform(54.5, 58.0, n_records)
    longitudes = np.random.uniform(7.5, 15.5, n_records)
    
    # Generate vessel speeds (0-30 knots)
    speeds = np.random.exponential(8, n_records)
    speeds = np.clip(speeds, 0, 30)
    
    # Generate course over ground (0-360 degrees)
    courses = np.random.uniform(0, 360, n_records)
    
    # Generate vessel types
    vessel_types = np.random.choice(["Class A", "Class B"], n_records, p=[0.7, 0.3])
    
    # Create DataFrame
    df = pd.DataFrame({
        "MMSI": np.random.choice(mmsis, n_records),
        "SOG": speeds,
        "COG": courses,
        "Longitude": longitudes,
        "Latitude": latitudes,
        "# Timestamp": timestamps,
        "Type of mobile": vessel_types
    })
    
    return df


def download_from_url(url: str, output_file: str, max_retries: int = 3) -> bool:
    """
    Download file from URL with retry logic.
    
    Args:
        url: URL to download from
        output_file: Output file path
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            print(f"   Downloading {url} (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"   ✅ Downloaded: {output_file}")
            return True
            
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    return False


def create_download_script_for_week(start_date: str, output_script: str = "download_week.sh"):
    """
    Create a shell script to download AIS data for a week.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        output_script: Output shell script path
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    script_content = f"""#!/bin/bash
# Danish AIS Data Download Script
# Week starting {start_date}

echo "Downloading Danish AIS data for week starting {start_date}"
mkdir -p raw_data

echo "Note: Replace these URLs with actual Danish AIS data sources"
echo "You may need to register at https://web.ais.dk/aisdata/"

"""
    
    for i in range(7):
        current_date = start_dt + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Example URLs (these would need to be updated with actual sources)
        script_content += f"""
echo "Downloading data for {date_str}..."
# Example URLs - replace with actual sources:
# wget -O raw_data/aisdk_{date_str}.csv "https://web.ais.dk/aisdata/aisdk_{date_str}.csv"
# OR if using API:
# curl -o raw_data/aisdk_{date_str}.csv "https://api.example.com/ais/data?date={date_str}&format=csv"

# For now, we'll use a placeholder
if [ ! -f raw_data/aisdk_{date_str}.csv ]; then
    echo "Please manually download aisdk_{date_str}.csv from the Danish AIS portal"
    echo "Place it in the raw_data/ directory"
fi
"""
    
    script_content += """
echo "Download script created!"
echo "Please update the URLs with actual Danish AIS data sources."
"""
    
    with open(output_script, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(output_script, 0o755)
    
    print(f"✅ Created download script: {output_script}")


def main():
    """Main function demonstrating data access options"""
    print("🚢 Danish AIS Data Access Guide")
    print("=" * 50)
    
    # Get data source information
    sources = get_danish_ais_urls()
    
    print("\n📡 Primary Data Source:")
    print(f"   Name: {sources['dma_primary']['name']}")
    print(f"   URL: {sources['dma_primary']['base_url']}")
    print(f"   Registration Required: {sources['dma_primary']['requires_registration']}")
    
    print("\n🔗 Alternative Sources:")
    for source in sources['alternative_sources']:
        print(f"   - {source['name']}: {source['url']}")
        print(f"     Note: {source['note']}")
    
    print("\n📋 Recommended Steps to Get Danish AIS Data:")
    print("   1. Visit https://web.ais.dk/aisdata/")
    print("   2. Register for access (if required)")
    print("   3. Browse available datasets")
    print("   4. Download CSV files for desired time period")
    print("   5. Place files in raw_data/ directory")
    
    print("\n🛠️  Example: Getting one week of data")
    print("   # Create download script for a week")
    print("   python scripts/download_danish_ais.py")
    
    # Example: Create download script for a week
    start_date = "2025-02-12"
    create_download_script_for_week(start_date)
    
    print(f"\n📅 Generating sample data for week starting {start_date}...")
    sample_files = generate_sample_data_for_week(start_date, "sample_data/")
    
    print(f"\n✅ Sample data generated for testing:")
    for file in sample_files:
        print(f"   {file}")
    
    print(f"\n🎉 Setup complete!")
    print(f"   - Use the generated download script to get real data")
    print(f"   - Or use the sample data for immediate testing")
    print(f"   - Run the data processing pipeline to convert to NPZ format")


if __name__ == "__main__":
    main()