import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, train_test_split
import logging

logger = logging.getLogger(__name__)

def load_and_prepare_data(data_dir='data/raw/'):
    """
    Load and merge tabular data from multiple sources.
    
    Returns:
        dict: Contains merged dataframes and metadata
    """
    try:
        # Load data sources
        logger.info("Loading tabular data...")
        
        # Priority 1: Check for enriched KOI data with stellar parameters
        koi_enriched_path = f"{data_dir}/lighkurve_KOI_dataset_enriched.csv"
        koi_path = f"{data_dir}/lighkurve_KOI_dataset.csv"
        
        if os.path.exists(koi_enriched_path):
            # Load enriched KOI data with stellar parameters
            koi_df = pd.read_csv(koi_enriched_path)
            logger.info(f"Loaded {len(koi_df)} enriched KOI records with {koi_df.shape[1]} features")
        elif os.path.exists(koi_path):
            # Load real KOI data with actual stellar parameters
            koi_df = pd.read_csv(koi_path)
            
            # Apply column renaming for stellar parameters if present
            rename_map = {
                'koi_teff': 'koi_steff',
                'teff': 'koi_steff',
                'logg': 'koi_slogg', 
                'feh': 'koi_smet',
                'stellar_radius': 'koi_srad',
                'stellar_mass': 'koi_smass',
                'transit_depth': 'koi_depth',
                'depth': 'koi_depth'
            }
            original_cols = set(koi_df.columns)
            koi_df = koi_df.rename(columns={k: v for k, v in rename_map.items() if k in koi_df.columns})
            renamed_cols = set(koi_df.columns) - original_cols
            if renamed_cols:
                logger.info(f"Renamed columns: {list(renamed_cols)}")
            
            logger.info(f"Loaded {len(koi_df)} real KOI records with {koi_df.shape[1]} features")
        else:
            # Priority 2: Check for minimal KOI labels file
            koi_labels_path = f"{data_dir}/q1_q17_dr25_sup_koi_2024.07.03_19.12.12.csv"
            global_path = f"{data_dir}/all_global.csv"
            local_path = f"{data_dir}/all_local.csv"
            
            # Strategy: Use synthetic global/local features if they exist
            if os.path.exists(global_path) and os.path.exists(local_path):
                # Load synthetic data format (global + local features)
                global_df = pd.read_csv(global_path)
                local_df = pd.read_csv(local_path) 
                
                # Merge features
                merged_df = global_df.merge(local_df, on='kepid', how='inner')
                
                # Load labels if available
                if os.path.exists(koi_labels_path):
                    koi_labels = pd.read_csv(koi_labels_path)
                    # Merge labels
                    merged_df = merged_df.merge(koi_labels[['kepid', 'koi_disposition']], on='kepid', how='left')
                    
                # Convert to KOI format for processing
                koi_df = merged_df
                logger.info(f"Loaded {len(koi_df)} synthetic records")
                
            elif os.path.exists(koi_labels_path):
                # Fall back to minimal labels only
                koi_df = pd.read_csv(koi_labels_path)
                logger.info(f"Loaded {len(koi_df)} minimal KOI records")
            else:
                logger.warning("No data files found")
                koi_df = pd.DataFrame()
        
        # For now, create empty dataframes for other sources
        toi_df = pd.DataFrame()
        tres_df = pd.DataFrame() 
        lk_df = pd.DataFrame()
        
        # Process and standardize column names
        processed_data = process_tabular_features(koi_df, toi_df, tres_df, lk_df)
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def process_tabular_features(koi_df, toi_df, tres_df, lk_df):
    """
    Process and standardize tabular features across datasets.
    
    Returns:
        dict: Processed dataframes with standardized features
    """
    features = []
    
    # Define common features to extract
    common_features = [
        'period', 'period_err1', 'period_err2',
        'epoch', 'epoch_err1', 'epoch_err2', 
        'depth', 'depth_err1', 'depth_err2',
        'duration', 'duration_err1', 'duration_err2',
        'impact', 'impact_err1', 'impact_err2',
        'star_temp', 'star_temp_err1', 'star_temp_err2',
        'star_radius', 'star_radius_err1', 'star_radius_err2',
        'star_mass', 'star_mass_err1', 'star_mass_err2',
        'star_logg', 'star_logg_err1', 'star_logg_err2',
        'star_metallicity', 'star_metallicity_err1', 'star_metallicity_err2'
    ]
    
    # Process KOI data
    if not koi_df.empty:
        koi_processed = extract_koi_features(koi_df)
        features.append(('koi', koi_processed))
    
    # Process TOI data  
    if not toi_df.empty:
        toi_processed = extract_toi_features(toi_df)
        features.append(('toi', toi_processed))
    
    # Process TrES data
    if not tres_df.empty:
        tres_processed = extract_tres_features(tres_df)
        features.append(('tres', tres_processed))
    
    # Process Lightkurve data
    if not lk_df.empty:
        lk_processed = extract_lightkurve_features(lk_df)
        features.append(('lightkurve', lk_processed))
    
    return {'features': features, 'common_columns': common_features}

def extract_koi_features(df):
    """Extract and standardize features from KOI dataset."""
    logger.info("Processing KOI features...")
    
    # Create target variable from disposition - use only CONFIRMED vs FALSE POSITIVE
    if 'koi_disposition' in df.columns:
        disp = df['koi_disposition'].astype(str).str.upper().str.strip()
        keep = disp.isin(['CONFIRMED', 'FALSE POSITIVE'])
        df = df.loc[keep].copy()
        df['is_planet'] = (disp.loc[keep] == 'CONFIRMED').astype(int)
        logger.info(f"Filtered to {len(df)} samples with clear labels (CONFIRMED/FALSE POSITIVE only)")
    else:
        # Handle synthetic data without disposition
        df['is_planet'] = 0  # Default to non-planet
    
    # For synthetic data, create dummy features if real KOI columns don't exist
    feature_mapping = {
        'koi_period': 'period',
        'koi_period_err1': 'period_err1', 
        'koi_period_err2': 'period_err2',
        'koi_time0bk': 'epoch',
        'koi_time0bk_err1': 'epoch_err1',
        'koi_time0bk_err2': 'epoch_err2',
        'koi_depth': 'depth',
        'koi_depth_err1': 'depth_err1',
        'koi_depth_err2': 'depth_err2',
        'koi_duration': 'duration', 
        'koi_duration_err1': 'duration_err1',
        'koi_duration_err2': 'duration_err2',
        'koi_impact': 'impact',
        'koi_impact_err1': 'impact_err1',
        'koi_impact_err2': 'impact_err2',
        'koi_slogg': 'star_logg',
        'koi_slogg_err1': 'star_logg_err1',
        'koi_slogg_err2': 'star_logg_err2',
        'koi_smet': 'star_metallicity',
        'koi_smet_err1': 'star_metallicity_err1',
        'koi_smet_err2': 'star_metallicity_err2',
        'koi_steff': 'star_temp',
        'koi_steff_err1': 'star_temp_err1',
        'koi_steff_err2': 'star_temp_err2',
        'koi_srad': 'star_radius',
        'koi_srad_err1': 'star_radius_err1',
        'koi_srad_err2': 'star_radius_err2',
        'koi_smass': 'star_mass',
        'koi_smass_err1': 'star_mass_err1',
        'koi_smass_err2': 'star_mass_err2',
    }
    
    # Check if we have real KOI columns or synthetic data
    has_real_koi_columns = any(col in df.columns for col in feature_mapping.keys())
    
    if has_real_koi_columns:
        # Extract real KOI features using the mapping
        extracted = pd.DataFrame()
        for old_name, new_name in feature_mapping.items():
            if old_name in df.columns:
                extracted[new_name] = df[old_name]
                
        # Also include any other numeric columns not in the mapping
        other_numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in other_numeric_cols:
            if col not in ['kepid'] and col not in feature_mapping.keys():
                extracted[col] = df[col]
                
        # 🔥 FEATURE ENGINEERING: Transit geometry & error ratios
        eps = 1e-12
        
        # Base columns (exist in your CSV)
        P = extracted.get('period', pd.Series(dtype=float))
        Pe1, Pe2 = extracted.get('period_err1', pd.Series(dtype=float)), extracted.get('period_err2', pd.Series(dtype=float))
        T0 = extracted.get('epoch', pd.Series(dtype=float))
        Te1, Te2 = extracted.get('epoch_err1', pd.Series(dtype=float)), extracted.get('epoch_err2', pd.Series(dtype=float))
        D = extracted.get('duration', pd.Series(dtype=float))
        De1, De2 = extracted.get('duration_err1', pd.Series(dtype=float)), extracted.get('duration_err2', pd.Series(dtype=float))
        
        if not P.empty and not D.empty:
            # Duty cycle + log scales
            extracted['duty_cycle'] = (D / (P + eps)).clip(lower=0)
            extracted['log_period'] = np.log10(P + eps)
            extracted['log_duration'] = np.log10(D + eps)
            
            # Symmetry & relative-uncertainty proxies
            if not Pe1.empty and not Pe2.empty:
                extracted['period_err_rel'] = (np.abs(Pe1) + np.abs(Pe2)) / (np.abs(P) + eps)
                extracted['err_asym_period'] = np.abs(np.abs(Pe1) - np.abs(Pe2)) / (np.abs(Pe1) + np.abs(Pe2) + eps)
            
            if not De1.empty and not De2.empty:
                extracted['duration_err_rel'] = (np.abs(De1) + np.abs(De2)) / (np.abs(D) + eps)
                extracted['err_asym_duration'] = np.abs(np.abs(De1) - np.abs(De2)) / (np.abs(De1) + np.abs(De2) + eps)
            
            if not Te1.empty and not Te2.empty:
                extracted['epoch_err_span'] = (np.abs(Te1) + np.abs(Te2))
        
        # If koi_quarters exists: count of quarters as numeric signal
        if 'koi_quarters' in df.columns:
            extracted['n_quarters'] = df['koi_quarters'].astype(str).str.count('1')  # Count observation quarters
            
        logger.info(f"Added {7 + (1 if 'koi_quarters' in df.columns else 0)} engineered features")
                
    else:
        # Handle synthetic data - use existing columns as features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['kepid']]
        
        # Copy all feature columns at once, preserving the DataFrame structure
        extracted = df[feature_cols].copy()
    
    # Add identifiers
    extracted['target_id'] = df.get('kepid', df.index)
    extracted['koi_id'] = df.get('kepoi_name', '')
    extracted['is_planet'] = df['is_planet']
    extracted['source'] = 'koi'
    
    logger.info(f"Extracted {len(extracted)} KOI features with {extracted.shape[1]-4} feature columns")
    return extracted

def extract_toi_features(df):
    """Extract and standardize features from TOI dataset."""
    logger.info("Processing TOI features...")
    
    # Create target variable
    df['is_planet'] = df.get('TFOPWG Disposition', '').str.contains('CP|PC', na=False).astype(int)
    
    feature_mapping = {
        'Period (days)': 'period',
        'Period (days) err': 'period_err1',
        'Epoch (BJD)': 'epoch', 
        'Epoch (BJD) err': 'epoch_err1',
        'Depth (ppm)': 'depth',
        'Depth (ppm) err': 'depth_err1',
        'Duration (hours)': 'duration',
        'Duration (hours) err': 'duration_err1',
        'Stellar Eff Temp (K)': 'star_temp',
        'Stellar Eff Temp (K) err': 'star_temp_err1',
        'Stellar Radius (R_Sun)': 'star_radius', 
        'Stellar Radius (R_Sun) err': 'star_radius_err1',
        'Stellar Mass (M_Sun)': 'star_mass',
        'Stellar Mass (M_Sun) err': 'star_mass_err1',
        'Stellar log(g) (cgs)': 'star_logg',
        'Stellar log(g) (cgs) err': 'star_logg_err1',
    }
    
    extracted = pd.DataFrame()
    for old_name, new_name in feature_mapping.items():
        if old_name in df.columns:
            extracted[new_name] = df[old_name]
    
    extracted['target_id'] = df.get('TIC ID', df.index)
    extracted['toi_id'] = df.get('TOI', '')
    extracted['is_planet'] = df['is_planet']
    extracted['source'] = 'toi'
    
    logger.info(f"Extracted {len(extracted)} TOI features")
    return extracted

def extract_tres_features(df):
    """Extract and standardize features from TrES dataset.""" 
    logger.info("Processing TrES features...")
    
    # TrES are confirmed planets
    df['is_planet'] = 1
    
    # Map available features
    feature_mapping = {
        'pl_orbper': 'period',
        'pl_orbpererr1': 'period_err1',
        'pl_orbpererr2': 'period_err2', 
        'pl_tranmid': 'epoch',
        'pl_tranmiderr1': 'epoch_err1',
        'pl_tranmiderr2': 'epoch_err2',
        'pl_trandep': 'depth',
        'pl_trandepERR1': 'depth_err1',
        'pl_trandepERR2': 'depth_err2',
        'pl_trandur': 'duration',
        'pl_trandurerr1': 'duration_err1', 
        'pl_trandurerr2': 'duration_err2',
        'st_teff': 'star_temp',
        'st_tefferr1': 'star_temp_err1',
        'st_tefferr2': 'star_temp_err2',
        'st_rad': 'star_radius',
        'st_raderr1': 'star_radius_err1',
        'st_raderr2': 'star_radius_err2',
        'st_mass': 'star_mass', 
        'st_masserr1': 'star_mass_err1',
        'st_masserr2': 'star_mass_err2',
        'st_logg': 'star_logg',
        'st_loggerr1': 'star_logg_err1',
        'st_loggerr2': 'star_logg_err2',
        'st_met': 'star_metallicity',
        'st_meterr1': 'star_metallicity_err1',
        'st_meterr2': 'star_metallicity_err2',
    }
    
    extracted = pd.DataFrame()
    for old_name, new_name in feature_mapping.items():
        if old_name in df.columns:
            extracted[new_name] = df[old_name]
    
    extracted['target_id'] = df.get('hostname', df.index) 
    extracted['tres_id'] = df.get('pl_name', '')
    extracted['is_planet'] = df['is_planet']
    extracted['source'] = 'tres'
    
    logger.info(f"Extracted {len(extracted)} TrES features") 
    return extracted

def extract_lightkurve_features(df):
    """Extract features from Lightkurve processed dataset."""
    logger.info("Processing Lightkurve features...")
    
    # Assume processed features already exist
    extracted = df.copy()
    
    if 'is_planet' not in extracted.columns:
        extracted['is_planet'] = 0  # Default to non-planet
    
    extracted['source'] = 'lightkurve'
    
    logger.info(f"Extracted {len(extracted)} Lightkurve features")
    return extracted

def create_train_val_test_splits(data, val_size=0.2, test_size=0.2, random_state=42):
    """
    Create train/validation/test splits with proper scaling
    """
    all_X = []
    all_y = []
    all_groups = []
    all_feature_names = None  # Track feature names
    
    for source, df in data['features']:
        # Get feature columns (exclude metadata and target)
        feature_cols = [col for col in df.columns 
                       if col not in ['target_id', 'koi_id', 'is_planet', 'source'] 
                       and df[col].dtype in ['int64', 'float64']]
        
        # Capture feature names from first source
        if all_feature_names is None:
            all_feature_names = list(feature_cols)
            print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        
        X = df[feature_cols].values
        y = df['is_planet'].values
        
        # Use target_id for grouping (same star system)
        groups = df['target_id'].values
        
        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups)
    
    # Combine all sources
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    groups_combined = np.hstack(all_groups)
    
    print(f"Total combined data: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_combined.astype(int))}")
    
    # Group-aware split to avoid data leakage
    gkf = GroupKFold(n_splits=int(1/test_size))
    train_val_idx, test_idx = next(gkf.split(X_combined, y_combined, groups_combined))
    
    X_train_val, X_test = X_combined[train_val_idx], X_combined[test_idx]
    y_train_val, y_test = y_combined[train_val_idx], y_combined[test_idx]
    groups_train_val = groups_combined[train_val_idx]
    
    # Split train_val into train and validation
    val_size_adjusted = val_size / (1 - test_size)
    gkf_val = GroupKFold(n_splits=int(1/val_size_adjusted))
    train_idx, val_idx = next(gkf_val.split(X_train_val, y_train_val, groups_train_val))
    
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    
    # Scale features using only training data
    scaler = StandardScaler()
    
    # Handle NaN values before scaling
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    
    # Fit imputer and scaler on training data only
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Apply to validation and test sets
    X_val_imputed = imputer.transform(X_val)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print(f"Applied median imputation and scaling to handle {np.isnan(X_train).sum()} NaN values")
    
    splits = {
        'train': {'X': X_train_scaled, 'y': y_train},
        'val': {'X': X_val_scaled, 'y': y_val},
        'test': {'X': X_test_scaled, 'y': y_test},
        'feature_names': all_feature_names
    }
    
    return splits

if __name__ == "__main__":
    # Test the data loading pipeline
    logging.basicConfig(level=logging.INFO)
    
    try:
        data = load_and_prepare_data()
        splits = create_train_val_test_splits(data)
        print("Data loading pipeline test successful!")
        
    except Exception as e:
        print(f"Data loading pipeline test failed: {e}")