"""
CycloVision AI - Phase 1: NOAA IBTrACS Dataset Verification
Inspects the local NOAA IBTrACS North Indian Ocean dataset.
Reports exact counts, columns, missing values, date ranges, and quality metrics.
"""

import os
import pandas as pd
import numpy as np

DATA_PATH = "data/ibtracs/NOAA_IBTrACS_NI_Cyclone_Track_ML_Dataset.csv"

def inspect_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset file not found at {DATA_PATH}")
        return
        
    print(f"Loading dataset: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Storm track identifiers
    # Check for SID (Storm ID) and NAME
    sid_col = 'SID' if 'SID' in df.columns else None
    name_col = 'NAME' if 'NAME' in df.columns else None
    
    num_storms_sid = df['SID'].nunique() if sid_col else 0
    num_storms_name = df['NAME'].nunique() if name_col else 0
    
    # Date range
    time_col = None
    for c in ['ISO_TIME', 'DATE_UTC', 'time', 'date']:
        if c in df.columns:
            time_col = c
            break
            
    min_date = df[time_col].dropna().min() if time_col else "Unknown"
    max_date = df[time_col].dropna().max() if time_col else "Unknown"
    
    # Coordinates
    lat_col = 'LAT' if 'LAT' in df.columns else ('lat' if 'lat' in df.columns else None)
    lon_col = 'LON' if 'LON' in df.columns else ('lon' if 'lon' in df.columns else None)
    
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    
    valid_lat_mask = df[lat_col].between(-90, 90)
    valid_lon_mask = df[lon_col].between(-180, 180)
    valid_coords = (valid_lat_mask & valid_lon_mask).sum()
    
    # Wind and pressure columns
    wind_cols = [c for c in df.columns if 'WIND' in c.upper() or 'WMO_WIND' in c.upper()]
    pres_cols = [c for c in df.columns if 'PRES' in c.upper() or 'WMO_PRES' in c.upper()]
    
    print("\n" + "="*60)
    print("NOAA IBTrACS DATASET VERIFICATION RESULTS")
    print("="*60)
    print(f"Total Observations (rows):    {total_rows}")
    print(f"Total Columns:                {total_cols}")
    print(f"Unique Storm Tracks (by SID): {num_storms_sid}")
    print(f"Unique Storm Names:           {num_storms_name}")
    print(f"Date Range:                   {min_date} to {max_date}")
    print(f"Valid Coordinates:            {valid_coords} / {total_rows} ({valid_coords/total_rows*100:.2f}%)")
    print(f"Latitude Range:               {df[lat_col].min():.2f} to {df[lat_col].max():.2f}")
    print(f"Longitude Range:              {df[lon_col].min():.2f} to {df[lon_col].max():.2f}")
    print("\nWind Column Candidates:")
    for w in wind_cols:
        non_null = pd.to_numeric(df[w], errors='coerce').notnull().sum()
        print(f"  - {w:20s}: {non_null:6d} non-null ({non_null/total_rows*100:5.1f}%)")
        
    print("\nPressure Column Candidates:")
    for p in pres_cols:
        non_null = pd.to_numeric(df[p], errors='coerce').notnull().sum()
        print(f"  - {p:20s}: {non_null:6d} non-null ({non_null/total_rows*100:5.1f}%)")
        
    # Storm track observation length stats
    track_lengths = df.groupby('SID').size()
    print("\nStorm Track Observation Length Distribution:")
    print(f"  Min points per storm:    {track_lengths.min()}")
    print(f"  Max points per storm:    {track_lengths.max()}")
    print(f"  Mean points per storm:   {track_lengths.mean():.1f}")
    print(f"  Median points per storm: {track_lengths.median():.1f}")
    print(f"  Storms with >= 12 pts:   {(track_lengths >= 12).sum()} storms (needed for 6-past -> 6-future)")
    print(f"  Storms with >= 9 pts:    {(track_lengths >= 9).sum()} storms (needed for 6-past -> 3-future)")
    print("="*60 + "\n")

if __name__ == "__main__":
    inspect_dataset()
