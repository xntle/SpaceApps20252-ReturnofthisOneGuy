import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

def download_lightcurve(target_id: str, mission: str = 'Kepler'):
    """
    Download lightcurve data using Lightkurve.
    
    Args:
        target_id: Target identifier (KIC, TIC, etc.)
        mission: Mission name ('Kepler', 'TESS', etc.)
        
    Returns:
        LightCurve object or None if download fails
    """
    try:
        logger.info(f"Downloading {mission} lightcurve for {target_id}")
        
        if mission.lower() == 'kepler':
            search_result = lk.search_lightcurve(f"KIC {target_id}", mission='Kepler')
        elif mission.lower() == 'tess':
            search_result = lk.search_lightcurve(f"TIC {target_id}", mission='TESS')
        else:
            search_result = lk.search_lightcurve(target_id, mission=mission)
        
        if len(search_result) == 0:
            logger.warning(f"No lightcurve found for {target_id}")
            return None
            
        # Download and stitch multiple quarters/sectors
        lc_collection = search_result.download_all()
        if lc_collection is None:
            logger.warning(f"Failed to download lightcurve for {target_id}")
            return None
            
        # Stitch together and normalize
        lc = lc_collection.stitch().normalize()
        
        try:
            time_span = float((lc.time.max() - lc.time.min()).value)  # Get the numeric value
            logger.info(f"Downloaded lightcurve: {len(lc)} points, {time_span:.1f} days")
        except:
            logger.info(f"Downloaded lightcurve: {len(lc)} points")
        return lc
        
    except Exception as e:
        logger.error(f"Error downloading lightcurve for {target_id}: {e}")
        return None

def preprocess_lightcurve(lc, 
                         sigma_clip: float = 5.0,
                         window_length: int = 101):
    """
    Preprocess lightcurve: outlier removal, detrending.
    
    Args:
        lc: Input lightcurve
        sigma_clip: Sigma threshold for outlier removal
        window_length: Window length for Savitzky-Golay filter
        
    Returns:
        Preprocessed lightcurve
    """
    logger.info("Preprocessing lightcurve...")
    
    # Remove outliers
    lc_clean = lc.remove_outliers(sigma=sigma_clip)
    
    # Remove NaN values
    lc_clean = lc_clean.remove_nans()
    
    # Detrend using Savitzky-Golay filter
    if len(lc_clean) > window_length:
        try:
            lc_detrended = lc_clean.flatten(window_length=window_length, method='savgol')
        except TypeError:
            # Fallback for newer Lightkurve versions without method parameter
            lc_detrended = lc_clean.flatten(window_length=window_length)
    else:
        # Use simpler detrending for short lightcurves
        try:
            lc_detrended = lc_clean.flatten(window_length=min(51, len(lc_clean)//2), method='median')
        except TypeError:
            # Fallback for newer Lightkurve versions
            lc_detrended = lc_clean.flatten(window_length=min(51, len(lc_clean)//2))
    
    logger.info(f"Preprocessing complete: {len(lc_detrended)} points remaining")
    return lc_detrended

def extract_lightcurve_features(lc) -> Dict[str, float]:
    """
    Extract statistical features from lightcurve.
    
    Args:
        lc: Input lightcurve
        
    Returns:
        Dictionary of extracted features
    """
    flux = lc.flux.value
    time = lc.time.value
    
    features = {}
    
    # Basic statistics
    features['lc_mean'] = np.mean(flux)
    features['lc_std'] = np.std(flux)
    features['lc_var'] = np.var(flux)
    features['lc_skewness'] = pd.Series(flux).skew()
    features['lc_kurtosis'] = pd.Series(flux).kurtosis()
    
    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        features[f'lc_percentile_{p}'] = np.percentile(flux, p)
    
    # Range and interquartile range
    features['lc_range'] = np.max(flux) - np.min(flux)
    features['lc_iqr'] = np.percentile(flux, 75) - np.percentile(flux, 25)
    
    # Time series specific features
    features['lc_duration'] = np.max(time) - np.min(time)
    features['lc_cadence'] = np.median(np.diff(time))
    features['lc_n_points'] = len(flux)
    
    # Power spectral density features
    try:
        freqs, psd = signal.periodogram(flux)
        features['psd_peak_freq'] = freqs[np.argmax(psd[1:])] if len(psd) > 1 else 0
        features['psd_peak_power'] = np.max(psd[1:]) if len(psd) > 1 else 0
        features['psd_total_power'] = np.sum(psd)
    except:
        features['psd_peak_freq'] = 0
        features['psd_peak_power'] = 0
        features['psd_total_power'] = 0
    
    # Autocorrelation features
    try:
        autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first minimum
        first_min_idx = 1
        while first_min_idx < len(autocorr) - 1 and autocorr[first_min_idx] >= autocorr[first_min_idx + 1]:
            first_min_idx += 1
        
        features['autocorr_first_min'] = autocorr[first_min_idx] if first_min_idx < len(autocorr) else 0
        features['autocorr_lag_first_min'] = first_min_idx * np.median(np.diff(time))
        
    except:
        features['autocorr_first_min'] = 0
        features['autocorr_lag_first_min'] = 0
    
    return features

def create_residual_windows(lc, 
                          period: float, 
                          epoch: float, 
                          duration: float,
                          window_factor: float = 3.0) -> np.ndarray:
    """
    Create windows around predicted transit times with residuals.
    
    Args:
        lc: Lightcurve object
        period: Transit period in days
        epoch: Transit epoch (time of first transit)
        duration: Transit duration in days
        window_factor: Window size as multiple of transit duration
        
    Returns:
        Array of windowed residuals for CNN input
    """
    time = lc.time.value
    flux = lc.flux.value
    
    # Calculate transit times
    n_transits = int((time.max() - epoch) / period) + 1
    transit_times = epoch + np.arange(n_transits) * period
    
    # Filter transit times within observation range
    transit_times = transit_times[(transit_times >= time.min()) & (transit_times <= time.max())]
    
    window_size = duration * window_factor
    windows = []
    
    for t_transit in transit_times:
        # Find points within window
        mask = (time >= t_transit - window_size/2) & (time <= t_transit + window_size/2)
        
        if np.sum(mask) < 10:  # Skip if too few points
            continue
        
        window_time = time[mask]
        window_flux = flux[mask]
        
        # Interpolate to fixed grid
        fixed_time = np.linspace(window_time.min(), window_time.max(), 128)
        fixed_flux = np.interp(fixed_time, window_time, window_flux)
        
        # Subtract local trend (residuals)
        trend = signal.savgol_filter(fixed_flux, min(51, len(fixed_flux)//2*2+1), 3)
        residuals = fixed_flux - trend
        
        windows.append(residuals)
    
    if not windows:
        logger.warning("No valid windows found")
        return np.array([]).reshape(0, 128)
    
    return np.array(windows)

def bls_period_search(lc, 
                     period_range: Tuple[float, float] = (0.5, 50.0),
                     n_periods: int = 10000) -> Dict[str, float]:
    """
    Perform Box Least Squares period search.
    
    Args:
        lc: Input lightcurve
        period_range: Min and max periods to search (days)
        n_periods: Number of trial periods
        
    Returns:
        Dictionary with BLS results
    """
    logger.info(f"Running BLS period search from {period_range[0]} to {period_range[1]} days")
    
    time = lc.time.value
    flux = lc.flux.value
    
    # Create periods array
    periods = np.linspace(period_range[0], period_range[1], n_periods)
    
    try:
        # Run BLS
        bls = BoxLeastSquares(time, flux)
        bls_result = bls.power(periods)
        
        # Find best period
        best_idx = np.argmax(bls_result.power)
        best_period = bls_result.period[best_idx]
        best_power = bls_result.power[best_idx]
        
        # Get statistics for best period
        bls_stats = bls.compute_stats(best_period)
        
        results = {
            'bls_period': best_period,
            'bls_power': best_power,
            'bls_epoch': bls_stats['transit_time'],
            'bls_duration': bls_stats['duration'],
            'bls_depth': bls_stats['depth'],
            'bls_snr': bls_stats.get('snr', 0),
            'bls_log_likelihood': bls_stats.get('log_likelihood', 0)
        }
        
        # Additional periodogram statistics
        results['bls_power_mean'] = np.mean(bls_result.power)
        results['bls_power_std'] = np.std(bls_result.power)
        results['bls_power_max'] = np.max(bls_result.power)
        results['bls_power_ratio'] = best_power / np.median(bls_result.power)
        
        logger.info(f"BLS found period: {best_period:.3f} days, power: {best_power:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"BLS period search failed: {e}")
        return {
            'bls_period': 0,
            'bls_power': 0,
            'bls_epoch': 0,
            'bls_duration': 0,
            'bls_depth': 0,
            'bls_snr': 0,
            'bls_log_likelihood': 0,
            'bls_power_mean': 0,
            'bls_power_std': 0,
            'bls_power_max': 0,
            'bls_power_ratio': 1
        }

def fold_lightcurve(lc, period: float, epoch: float):
    """
    Fold lightcurve to given period and epoch.
    
    Args:
        lc: Input lightcurve
        period: Folding period in days
        epoch: Reference epoch
        
    Returns:
        Folded lightcurve
    """
    # Calculate phase
    phase = ((lc.time.value - epoch) % period) / period
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    
    # Create folded lightcurve
    folded_lc = lk.LightCurve(
        time=phase[sort_idx] * period + epoch,
        flux=lc.flux[sort_idx]
    )
    
    return folded_lc

def bin_lightcurve(lc, bin_size: float = 0.01):
    """
    Bin lightcurve for visualization and analysis.
    
    Args:
        lc: Input lightcurve
        bin_size: Bin size in same units as time
        
    Returns:
        Binned lightcurve
    """
    time = lc.time.value
    flux = lc.flux.value
    
    # Create bins
    time_min, time_max = time.min(), time.max()
    n_bins = int((time_max - time_min) / bin_size)
    bin_edges = np.linspace(time_min, time_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Bin the data
    binned_flux = []
    binned_err = []
    
    for i in range(len(bin_centers)):
        mask = (time >= bin_edges[i]) & (time < bin_edges[i + 1])
        if np.sum(mask) > 0:
            binned_flux.append(np.mean(flux[mask]))
            binned_err.append(np.std(flux[mask]) / np.sqrt(np.sum(mask)))
        else:
            binned_flux.append(np.nan)
            binned_err.append(np.nan)
    
    # Remove NaN values
    valid_mask = ~np.isnan(binned_flux)
    
    binned_lc = lk.LightCurve(
        time=bin_centers[valid_mask],
        flux=np.array(binned_flux)[valid_mask],
        flux_err=np.array(binned_err)[valid_mask]
    )
    
    return binned_lc

def process_target_lightcurve(target_id: str, 
                            mission: str = 'Kepler',
                            known_period: Optional[float] = None,
                            known_epoch: Optional[float] = None,
                            known_duration: Optional[float] = None) -> Dict:
    """
    Complete lightcurve processing pipeline for a single target.
    
    Args:
        target_id: Target identifier
        mission: Mission name
        known_period: Known period for residual windows (optional)
        known_epoch: Known epoch for residual windows (optional)
        known_duration: Known duration for residual windows (optional)
        
    Returns:
        Dictionary with all extracted features and data
    """
    logger.info(f"Processing target {target_id}")
    
    results = {'target_id': target_id, 'success': False}
    
    try:
        # Download lightcurve
        lc = download_lightcurve(target_id, mission)
        if lc is None:
            return results
        
        # Preprocess
        lc_clean = preprocess_lightcurve(lc)
        
        # Extract basic features
        lc_features = extract_lightcurve_features(lc_clean)
        results.update(lc_features)
        
        # BLS period search
        bls_results = bls_period_search(lc_clean)
        results.update(bls_results)
        
        # Use known parameters if available, otherwise BLS results
        period = known_period if known_period else bls_results['bls_period']
        epoch = known_epoch if known_epoch else bls_results['bls_epoch']
        duration = known_duration if known_duration else bls_results['bls_duration']
        
        # Create residual windows if we have valid parameters
        if period > 0 and duration > 0:
            residual_windows = create_residual_windows(lc_clean, period, epoch, duration)
            results['residual_windows'] = residual_windows
            results['n_windows'] = len(residual_windows)
        else:
            results['residual_windows'] = np.array([]).reshape(0, 128)
            results['n_windows'] = 0
        
        # Store lightcurve data
        results['lightcurve'] = lc_clean
        results['success'] = True
        
        logger.info(f"Successfully processed target {target_id}")
        
    except Exception as e:
        logger.error(f"Error processing target {target_id}: {e}")
    
    return results

if __name__ == "__main__":
    # Test the feature extraction pipeline
    logging.basicConfig(level=logging.INFO)
    
    # Test with a known Kepler target
    test_target = "757076"  # Kepler-10
    
    results = process_target_lightcurve(test_target)
    
    if results['success']:
        print("Feature extraction test successful!")
        print(f"Extracted {len([k for k in results.keys() if k.startswith('lc_')])} lightcurve features")
        print(f"Extracted {len([k for k in results.keys() if k.startswith('bls_')])} BLS features")
        print(f"Created {results['n_windows']} residual windows")
    else:
        print("Feature extraction test failed!")