# Danish AIS Data Access Guide

## 📡 Overview

This guide explains how to access and download AIS (Automatic Identification System) data from Danish sources for vessel trajectory prediction.

## 🔗 Primary Data Sources

### 1. Danish Maritime Authority (DMA)
**URL**: https://web.ais.dk/aisdata/
- **Official Danish AIS data portal**
- Requires registration (free)
- Provides historical AIS data
- Data format: CSV files with standard AIS fields

### 2. Alternative Sources
- **MarineTraffic**: https://www.marinetraffic.com/en/data/historical-data
- **AISHub**: https://www.aishub.net/
- **VesselFinder**: Historical data services

## 📋 Step-by-Step Download Process

### Step 1: Registration
1. Visit https://web.ais.dk/aisdata/
2. Register for a user account (free)
3. Verify your email address
4. Log in to access the portal

### Step 2: Browse Available Data
1. Navigate to "Historical Data" or "Data Downloads"
2. Select your desired time period
3. Choose data format (CSV recommended)
4. Check data availability and size

### Step 3: Download Data
1. Select multiple days for extended datasets
2. Download files to your local machine
3. Organize files by date (e.g., `aisdk_2025-02-12.csv`)
4. Verify file integrity after download

## 🗓️ Getting Multiple Days of Data

### Method 1: Manual Download
```bash
# Download files manually and organize them
mkdir -p raw_data
# Place downloaded files: aisdk_YYYY-MM-DD.csv
```

### Method 2: Automated Download Script
```bash
# Generate download script for a week
python scripts/download_danish_ais.py
# This creates download_week.sh with example URLs
```

### Method 3: Use Provided Sample Data
```bash
# Generate sample data for testing
python scripts/download_danish_ais.py
# This creates sample data files for immediate use
```

## 📊 Data Format

### CSV Structure
The Danish AIS data typically includes these fields:
- **MMSI**: Maritime Mobile Service Identity (vessel ID)
- **SOG**: Speed Over Ground (knots)
- **COG**: Course Over Ground (degrees)
- **Longitude**: Position longitude
- **Latitude**: Position latitude
- **Timestamp**: Date and time
- **Type of mobile**: Vessel class (Class A/Class B)

### Example Data Processing
```python
# Process multiple CSV files
python main.py --csv_files raw_data/*.csv --process_data --epochs 30
```

## 🌍 Geographic Coverage

### Denmark Region Bounds
- **North**: 58.0°N
- **South**: 54.5°N
- **West**: 7.5°E
- **East**: 15.5°E

### Maritime Identification Digits (MID) for Denmark
- Primary MIDs: 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
- Additional MIDs: 219, 220

## ⚙️ Data Processing Pipeline

### Complete Pipeline Usage
```python
from data_processing import run_complete_pipeline

# Process a week of CSV files
results = run_complete_pipeline(
    csv_files=[
        "raw_data/aisdk_2025-02-12.csv",
        "raw_data/aisdk_2025-02-13.csv",
        "raw_data/aisdk_2025-02-14.csv"
        # ... more files
    ],
    output_base_dir="processed_ais_data",
    resample_interval="1min",
    max_seq_len=256,
    bbox=[58.0, 7.5, 54.5, 15.5]  # Denmark region
)
```

## 🛠️ Technical Requirements

### Dependencies
```bash
pip install pandas pyarrow numpy requests tqdm
```

### Storage Requirements
- Single day: ~50-200 MB (compressed)
- One week: ~350 MB - 1.4 GB
- One month: ~1.5 - 6 GB

## 📈 Data Quality Considerations

### Filtering Criteria
- **Track Length**: Minimum 256 positions
- **Speed Range**: 1-50 knots (filters stationary objects)
- **Time Span**: Minimum 1 hour duration
- **Position Validity**: Within geographic bounds
- **MMSI Validation**: Standard 9-digit format

### Data Cleaning
- Remove duplicate positions
- Filter invalid coordinates
- Validate timestamps
- Segment tracks by time gaps (>15 minutes)

## 🎯 Usage Examples

### Basic Usage
```bash
# Process single file
python data_processing/pipeline.py --csv_file aisdk_2025-02-12.csv

# Process multiple files
python main.py --csv_files *.csv --process_data

# Use existing NPZ file
python main.py --data_path processed_data.npz
```

### Advanced Usage
```bash
# Process with custom parameters
python main.py \
    --csv_files raw_data/*.csv \
    --process_data \
    --resample_interval 30s \
    --max_seq_len 512 \
    --bbox 56.0 10.0 55.0 12.0 \
    --epochs 50 \
    --output_dir results/
```

## 🔍 Troubleshooting

### Common Issues
1. **Registration Required**: Ensure you have valid login credentials
2. **Large File Sizes**: Use download managers for big datasets
3. **Data Gaps**: Some days may have limited coverage
4. **Format Changes**: Verify CSV structure matches expectations

### Solutions
- Contact DMA support for access issues
- Use sample data for initial testing
- Implement error handling in processing scripts
- Validate data format before processing

## 📚 Additional Resources

### Documentation
- [Danish AIS Portal Documentation](https://web.ais.dk/aisdata/)
- [AIS Data Format Specifications](https://www.itu.int/rec/R-REC-M.1371)
- [Maritime Identification Digits](https://www.itu.int/en/ITU-R/terrestrial/fmd/Pages/mid.aspx)

### Related Tools
- [AIS Decoder Tools](https://www.aishub.net/ais-decoder)
- [Maritime Data Standards](https://www.iho.int/)

## 🎉 Next Steps

1. **Register** at the Danish AIS portal
2. **Download** data for your desired time period
3. **Process** data using the provided pipeline
4. **Train** models with the processed data
5. **Visualize** results using the enhanced framework

## 📞 Support

For data access issues:
1. Contact Danish Maritime Authority support
2. Check portal documentation
3. Use sample data for testing
4. Implement error handling in your scripts

---

*This guide is regularly updated. Please check the latest version of the Danish AIS portal for current procedures and requirements.*