#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def load_cao_dataset(excel_path='data/41586_2022_5644_MOESM4_ESM.xlsx'):
    """
    Load and preprocess Cao et al. dataset (Nature 2022)
    
    This dataset contains:
    - Antibody sequences (VH and VL chains)
    - Epitope group classifications
    - Neutralization data
    
    Returns:
        pd.DataFrame: Processed dataset with VH/VL sequences and metadata
    """
    print("="*80)
    print("LOADING CAO ET AL. DATASET (Nature 2022)")
    print("="*80)
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Cao dataset not found at {excel_path}")
    
    # Read Excel file - the first row contains the actual headers
    print(f"Reading Excel file: {excel_path}")
    df_raw = pd.read_excel(excel_path)
    
    # Extract headers from first row and use as column names
    headers = df_raw.iloc[0].tolist()
    df = pd.DataFrame(df_raw.iloc[1:].values, columns=headers)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean up column names
    df.columns = [str(col).strip() if col is not None else f'col_{i}' 
                  for i, col in enumerate(df.columns)]
    
    # Identify key columns
    antibody_col = None
    epitope_col = None
    heavy_chain_col = None
    light_chain_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'antibody' in col_lower and 'name' in col_lower:
            antibody_col = col
        elif 'epitope' in col_lower and 'group' in col_lower:
            epitope_col = col
        elif 'heavy' in col_lower and ('chain' in col_lower or 'aa' in col_lower):
            heavy_chain_col = col
        elif 'light' in col_lower and ('chain' in col_lower or 'aa' in col_lower):
            light_chain_col = col
    
    print(f"\nIdentified key columns:")
    print(f"  Antibody: {antibody_col}")
    print(f"  Epitope Group: {epitope_col}")
    print(f"  Heavy Chain: {heavy_chain_col}")
    print(f"  Light Chain: {light_chain_col}")
    
    # Verify we have the essential columns
    if not all([antibody_col, epitope_col, heavy_chain_col, light_chain_col]):
        print("\nColumn detection failed. Showing first few rows for manual inspection:")
        print(df.head())
        raise ValueError("Could not identify essential columns in the dataset")
    
    # Extract and clean the data
    processed_data = []
    
    for idx, row in df.iterrows():
        antibody_name = str(row[antibody_col]).strip()
        epitope_group = str(row[epitope_col]).strip()
        heavy_seq = str(row[heavy_chain_col]).strip()
        light_seq = str(row[light_chain_col]).strip()
        
        # Skip rows with missing essential data
        if (antibody_name in ['nan', 'None', ''] or 
            epitope_group in ['nan', 'None', ''] or
            heavy_seq in ['nan', 'None', ''] or
            light_seq in ['nan', 'None', '']):
            continue
        
        # Basic sequence validation
        if len(heavy_seq) < 50 or len(light_seq) < 50:  # Minimum reasonable length
            continue
        
        processed_data.append({
            'antibody_id': antibody_name,
            'epitope_group': epitope_group,
            'heavy_chain_aa': heavy_seq,
            'light_chain_aa': light_seq,
            'source': 'cao_nature_2022'
        })
    
    processed_df = pd.DataFrame(processed_data)
    
    print(f"\nProcessed dataset:")
    print(f"  Total antibodies: {len(processed_df)}")
    print(f"  Epitope groups: {processed_df['epitope_group'].nunique()}")
    
    # Show epitope distribution
    epitope_counts = processed_df['epitope_group'].value_counts()
    print(f"\nEpitope group distribution:")
    for epitope, count in epitope_counts.items():
        print(f"  {epitope}: {count} antibodies")
    
    # Show sequence length statistics
    heavy_lengths = processed_df['heavy_chain_aa'].str.len()
    light_lengths = processed_df['light_chain_aa'].str.len()
    
    print(f"\nSequence length statistics:")
    print(f"  Heavy chain - Min: {heavy_lengths.min()}, Max: {heavy_lengths.max()}, Mean: {heavy_lengths.mean():.1f}")
    print(f"  Light chain - Min: {light_lengths.min()}, Max: {light_lengths.max()}, Mean: {light_lengths.mean():.1f}")
    
    return processed_df

def load_neutralization_data(csv_path='data/src_neut_data.csv'):
    """
    Load neutralization data for evaluation labels

    Returns:
        pd.DataFrame: Neutralization IC50 data
    """
    print(f"\nLoading neutralization data: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"Warning: Neutralization data not found at {csv_path}")
        return None

    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                neut_df = pd.read_csv(csv_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"Warning: Could not decode {csv_path} with any standard encoding")
            return None

        print(f"Neutralization data shape: {neut_df.shape}")
        print(f"Columns: {list(neut_df.columns)}")

        return neut_df

    except Exception as e:
        print(f"Warning: Error loading neutralization data: {e}")
        return None

def integrate_datasets(cao_df, neut_df=None):
    """
    Integrate Cao dataset with neutralization data
    
    Args:
        cao_df: Processed Cao dataset
        neut_df: Neutralization data (optional)
    
    Returns:
        pd.DataFrame: Integrated dataset ready for structure embedding
    """
    print(f"\nIntegrating datasets...")
    
    integrated_df = cao_df.copy()
    
    if neut_df is not None:
        # Merge on antibody name/ID
        antibody_col = 'antibody' if 'antibody' in neut_df.columns else neut_df.columns[0]
        
        # Create mapping from neutralization data
        neut_mapping = {}
        for _, row in neut_df.iterrows():
            ab_name = str(row[antibody_col]).strip()
            neut_mapping[ab_name] = {
                'D614G_IC50': row.get('D614G_IC50', np.nan),
                'BA1_IC50': row.get('BA1_IC50', np.nan),
                'BA2_IC50': row.get('BA2_IC50', np.nan),
                'BA5_IC50': row.get('BA5_IC50', np.nan)
            }
        
        # Add neutralization data to integrated dataset
        for col in ['D614G_IC50', 'BA1_IC50', 'BA2_IC50', 'BA5_IC50']:
            integrated_df[col] = integrated_df['antibody_id'].map(
                lambda x: neut_mapping.get(x, {}).get(col, np.nan)
            )
        
        matched = integrated_df['D614G_IC50'].notna().sum()
        print(f"  Matched {matched}/{len(integrated_df)} antibodies with neutralization data")
    
    print(f"Final integrated dataset shape: {integrated_df.shape}")
    
    return integrated_df

def main():
    """Main function for data preprocessing"""
    print("Starting data preprocessing for structure embedding pipeline...")
    
    # Load Cao dataset
    cao_df = load_cao_dataset()
    
    # Load neutralization data
    neut_df = load_neutralization_data()
    
    # Integrate datasets
    integrated_df = integrate_datasets(cao_df, neut_df)
    
    # Save processed data
    output_path = 'data/processed_cao_dataset.csv'
    integrated_df.to_csv(output_path, index=False)
    print(f"\nSaved processed dataset to: {output_path}")
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Ready for structure embedding pipeline:")
    print(f"  - {len(integrated_df)} antibodies with VH/VL sequences")
    print(f"  - {integrated_df['epitope_group'].nunique()} epitope groups")
    print(f"  - Ready for IgFold structure generation")
    
    return integrated_df

if __name__ == "__main__":
    main()
