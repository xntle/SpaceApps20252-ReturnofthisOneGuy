import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import lightkurve as lk
from scipy import ndimage
from skimage import measure
import logging
from typing import Tuple, Dict, Optional, List

logger = logging.getLogger(__name__)

def download_target_pixel_file(target_id: str, mission: str = 'Kepler'):
    """
    Download Target Pixel File (TPF) using Lightkurve.
    
    Args:
        target_id: Target identifier (KIC, TIC, etc.)
        mission: Mission name ('Kepler', 'TESS', etc.)
        
    Returns:
        TargetPixelFile object or None if download fails
    """
    try:
        logger.info(f"Downloading {mission} TPF for {target_id}")
        
        if mission.lower() == 'kepler':
            search_result = lk.search_targetpixelfile(f"KIC {target_id}", mission='Kepler')
        elif mission.lower() == 'tess':
            search_result = lk.search_targetpixelfile(f"TIC {target_id}", mission='TESS')
        else:
            search_result = lk.search_targetpixelfile(target_id, mission=mission)
        
        if len(search_result) == 0:
            logger.warning(f"No TPF found for {target_id}")
            return None
            
        # Download first available TPF
        tpf = search_result[0].download()
        if tpf is None:
            logger.warning(f"Failed to download TPF for {target_id}")
            return None
            
        logger.info(f"Downloaded TPF: {tpf.flux.shape} (time, row, col)")
        return tpf
        
    except Exception as e:
        logger.error(f"Error downloading TPF for {target_id}: {e}")
        return None

def create_aperture_mask(tpf, 
                        threshold_sigma: float = 3.0,
                        method: str = 'threshold') -> np.ndarray:
    """
    Create aperture mask for extracting lightcurve.
    
    Args:
        tpf: Target Pixel File
        threshold_sigma: Threshold in sigma above median for mask creation
        method: Method for mask creation ('threshold', 'pipeline', 'custom')
        
    Returns:
        Boolean mask array
    """
    if method == 'pipeline' and hasattr(tpf, 'pipeline_mask'):
        # Use pipeline mask if available
        return tpf.pipeline_mask
    
    elif method == 'threshold':
        # Create threshold-based mask
        median_image = np.median(tpf.flux.value, axis=0)
        mad = np.median(np.abs(median_image - np.median(median_image)))
        threshold = np.median(median_image) + threshold_sigma * 1.4826 * mad
        
        mask = median_image > threshold
        
        # Clean up mask - remove isolated pixels
        mask = ndimage.binary_opening(mask)
        mask = ndimage.binary_closing(mask)
        
        return mask
    
    else:
        # Custom method - use largest contiguous region
        median_image = np.median(tpf.flux.value, axis=0)
        threshold = np.percentile(median_image, 75)
        
        labeled, n_features = ndimage.label(median_image > threshold)
        if n_features > 0:
            # Find largest connected component
            sizes = [np.sum(labeled == i) for i in range(1, n_features + 1)]
            largest_label = np.argmax(sizes) + 1
            mask = labeled == largest_label
        else:
            # Fallback to threshold method
            mask = median_image > threshold
        
        return mask

def compute_pixel_differences(tpf, 
                            period: float, 
                            epoch: float, 
                            duration: float,
                            phase_bins: int = 32) -> np.ndarray:
    """
    Compute pixel-level differences between in-transit and out-of-transit images.
    
    Args:
        tpf: Target Pixel File
        period: Transit period in days
        epoch: Transit epoch (time of first transit) 
        duration: Transit duration in days
        phase_bins: Number of phase bins for folding
        
    Returns:
        Array of pixel difference images
    """
    logger.info("Computing pixel differences...")
    
    time = tpf.time.value
    flux = tpf.flux.value
    
    # Remove NaN cadences
    valid_mask = np.isfinite(time) & np.all(np.isfinite(flux.reshape(len(flux), -1)), axis=1)
    time = time[valid_mask]
    flux = flux[valid_mask]
    
    if len(time) == 0:
        logger.warning("No valid cadences found")
        return np.array([])
    
    # Calculate phase
    phase = ((time - epoch) % period) / period
    
    # Define in-transit and out-of-transit phases
    transit_phase_width = duration / period
    in_transit_mask = (phase < transit_phase_width/2) | (phase > 1 - transit_phase_width/2)
    
    # Out-of-transit: phases around 0.25 and 0.75 (quadrature)
    out_transit_mask = ((phase > 0.2) & (phase < 0.3)) | ((phase > 0.7) & (phase < 0.8))
    
    if np.sum(in_transit_mask) == 0 or np.sum(out_transit_mask) == 0:
        logger.warning("Insufficient in-transit or out-of-transit data")
        return np.array([])
    
    # Compute median images
    in_transit_image = np.median(flux[in_transit_mask], axis=0)
    out_transit_image = np.median(flux[out_transit_mask], axis=0)
    
    # Compute difference
    diff_image = out_transit_image - in_transit_image
    
    # Normalize difference by out-of-transit flux
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_diff = diff_image / out_transit_image
        normalized_diff = np.nan_to_num(normalized_diff, 0)
    
    # Create phase-binned differences for time series analysis
    phase_bin_edges = np.linspace(0, 1, phase_bins + 1)
    phase_bin_centers = (phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2
    
    binned_diffs = []
    
    for i in range(phase_bins):
        bin_mask = (phase >= phase_bin_edges[i]) & (phase < phase_bin_edges[i + 1])
        
        if np.sum(bin_mask) > 0:
            bin_image = np.median(flux[bin_mask], axis=0)
            bin_diff = bin_image - out_transit_image
            
            with np.errstate(divide='ignore', invalid='ignore'):
                bin_diff_norm = bin_diff / out_transit_image
                bin_diff_norm = np.nan_to_num(bin_diff_norm, 0)
            
            binned_diffs.append(bin_diff_norm)
        else:
            # Use zeros for empty bins
            binned_diffs.append(np.zeros_like(out_transit_image))
    
    binned_diffs = np.array(binned_diffs)
    
    logger.info(f"Computed pixel differences: {binned_diffs.shape}")
    return binned_diffs

def extract_pixel_features(tpf, aperture_mask: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from pixel data.
    
    Args:
        tpf: Target Pixel File
        aperture_mask: Boolean mask for aperture
        
    Returns:
        Dictionary of pixel-based features
    """
    flux = tpf.flux.value
    time = tpf.time.value
    
    # Remove NaN cadences
    valid_mask = np.isfinite(time) & np.all(np.isfinite(flux.reshape(len(flux), -1)), axis=1)
    flux = flux[valid_mask]
    
    if len(flux) == 0:
        logger.warning("No valid cadences for feature extraction")
        return {}
    
    features = {}
    
    # Median image statistics
    median_image = np.median(flux, axis=0)
    features['pixel_median_total'] = np.sum(median_image)
    features['pixel_median_aperture'] = np.sum(median_image[aperture_mask])
    features['pixel_median_background'] = np.sum(median_image[~aperture_mask])
    
    # Image shape and centroid
    features['pixel_aperture_size'] = np.sum(aperture_mask)
    features['pixel_total_size'] = aperture_mask.size
    features['pixel_aperture_fraction'] = np.sum(aperture_mask) / aperture_mask.size
    
    # Compute centroid over time
    centroids_row = []
    centroids_col = []
    
    for i in range(len(flux)):
        img = flux[i]
        if np.any(np.isfinite(img)):
            # Use aperture region for centroid calculation
            aperture_img = img * aperture_mask
            if np.sum(aperture_img) > 0:
                row_coords, col_coords = np.mgrid[0:img.shape[0], 0:img.shape[1]]
                centroid_row = np.sum(aperture_img * row_coords) / np.sum(aperture_img)
                centroid_col = np.sum(aperture_img * col_coords) / np.sum(aperture_img)
                centroids_row.append(centroid_row)
                centroids_col.append(centroid_col)
    
    if centroids_row:
        features['pixel_centroid_row_mean'] = np.mean(centroids_row)
        features['pixel_centroid_col_mean'] = np.mean(centroids_col)
        features['pixel_centroid_row_std'] = np.std(centroids_row)
        features['pixel_centroid_col_std'] = np.std(centroids_col)
        features['pixel_centroid_row_range'] = np.max(centroids_row) - np.min(centroids_row)
        features['pixel_centroid_col_range'] = np.max(centroids_col) - np.min(centroids_col)
    else:
        features['pixel_centroid_row_mean'] = 0
        features['pixel_centroid_col_mean'] = 0
        features['pixel_centroid_row_std'] = 0
        features['pixel_centroid_col_std'] = 0
        features['pixel_centroid_row_range'] = 0
        features['pixel_centroid_col_range'] = 0
    
    # Flux variability per pixel
    aperture_pixels = flux[:, aperture_mask]
    if aperture_pixels.size > 0:
        pixel_stds = np.std(aperture_pixels, axis=0)
        pixel_means = np.mean(aperture_pixels, axis=0)
        
        features['pixel_std_mean'] = np.mean(pixel_stds)
        features['pixel_std_max'] = np.max(pixel_stds)
        features['pixel_mean_min'] = np.min(pixel_means)
        features['pixel_mean_max'] = np.max(pixel_means)
        features['pixel_snr_mean'] = np.mean(pixel_means / pixel_stds) if np.all(pixel_stds > 0) else 0
    else:
        features['pixel_std_mean'] = 0
        features['pixel_std_max'] = 0
        features['pixel_mean_min'] = 0
        features['pixel_mean_max'] = 0
        features['pixel_snr_mean'] = 0
    
    # Cross-pixel correlations
    if aperture_pixels.shape[1] > 1:
        corr_matrix = np.corrcoef(aperture_pixels.T)
        corr_matrix = corr_matrix[~np.isnan(corr_matrix)]
        if len(corr_matrix) > 0:
            features['pixel_corr_mean'] = np.mean(corr_matrix)
            features['pixel_corr_std'] = np.std(corr_matrix)
        else:
            features['pixel_corr_mean'] = 0
            features['pixel_corr_std'] = 0
    else:
        features['pixel_corr_mean'] = 0
        features['pixel_corr_std'] = 0
    
    return features

def create_pixel_lightcurve(tpf, aperture_mask: np.ndarray):
    """
    Create lightcurve from TPF using aperture mask.
    
    Args:
        tpf: Target Pixel File
        aperture_mask: Boolean mask for aperture
        
    Returns:
        LightCurve object
    """
    # Sum flux within aperture
    aperture_flux = np.sum(tpf.flux.value[:, aperture_mask], axis=1)
    
    # Create lightcurve
    lc = lk.LightCurve(
        time=tpf.time,
        flux=aperture_flux,
        flux_err=np.sqrt(aperture_flux),  # Approximation
        meta=tpf.meta
    )
    
    return lc

def analyze_crowding(tpf, aperture_mask: np.ndarray) -> Dict[str, float]:
    """
    Analyze crowding metrics from pixel data.
    
    Args:
        tpf: Target Pixel File
        aperture_mask: Boolean mask for aperture
        
    Returns:
        Dictionary of crowding metrics
    """
    features = {}
    
    median_image = np.median(tpf.flux.value, axis=0)
    
    # CROWDSAP - fraction of flux in aperture vs. total
    total_flux = np.sum(median_image)
    aperture_flux = np.sum(median_image[aperture_mask])
    
    if total_flux > 0:
        features['crowdsap'] = aperture_flux / total_flux
    else:
        features['crowdsap'] = 0
    
    # Flux concentration metrics
    sorted_pixels = np.sort(median_image.flatten())[::-1]  # Descending order
    
    # Fraction of flux in brightest N pixels
    for n in [1, 4, 9, 16]:
        if len(sorted_pixels) >= n:
            features[f'flux_frac_top_{n}'] = np.sum(sorted_pixels[:n]) / total_flux if total_flux > 0 else 0
        else:
            features[f'flux_frac_top_{n}'] = 0
    
    # Background estimation
    if aperture_mask.size > np.sum(aperture_mask):
        background_pixels = median_image[~aperture_mask]
        features['background_median'] = np.median(background_pixels)
        features['background_std'] = np.std(background_pixels)
        features['background_to_aperture_ratio'] = features['background_median'] / np.median(median_image[aperture_mask]) if np.median(median_image[aperture_mask]) > 0 else 0
    else:
        features['background_median'] = 0
        features['background_std'] = 0
        features['background_to_aperture_ratio'] = 0
    
    return features

def process_target_pixel_file(target_id: str, 
                            mission: str = 'Kepler',
                            known_period: Optional[float] = None,
                            known_epoch: Optional[float] = None,
                            known_duration: Optional[float] = None) -> Dict:
    """
    Complete TPF processing pipeline for a single target.
    
    Args:
        target_id: Target identifier
        mission: Mission name
        known_period: Known period for pixel differences (optional)
        known_epoch: Known epoch for pixel differences (optional)
        known_duration: Known duration for pixel differences (optional)
        
    Returns:
        Dictionary with all extracted features and data
    """
    logger.info(f"Processing TPF for target {target_id}")
    
    results = {'target_id': target_id, 'success': False}
    
    try:
        # Download TPF
        tpf = download_target_pixel_file(target_id, mission)
        if tpf is None:
            return results
        
        # Create aperture mask
        aperture_mask = create_aperture_mask(tpf)
        
        # Extract pixel features
        pixel_features = extract_pixel_features(tpf, aperture_mask)
        results.update(pixel_features)
        
        # Analyze crowding
        crowding_features = analyze_crowding(tpf, aperture_mask)
        results.update(crowding_features)
        
        # Create pixel lightcurve
        pixel_lc = create_pixel_lightcurve(tpf, aperture_mask)
        results['pixel_lightcurve'] = pixel_lc
        
        # Compute pixel differences if we have known parameters
        if all(x is not None for x in [known_period, known_epoch, known_duration]):
            pixel_diffs = compute_pixel_differences(tpf, known_period, known_epoch, known_duration)
            results['pixel_differences'] = pixel_diffs
        else:
            results['pixel_differences'] = np.array([])
        
        # Store TPF and mask
        results['tpf'] = tpf
        results['aperture_mask'] = aperture_mask
        results['success'] = True
        
        logger.info(f"Successfully processed TPF for target {target_id}")
        
    except Exception as e:
        logger.error(f"Error processing TPF for target {target_id}: {e}")
    
    return results

def save_pixel_differences(pixel_diffs: np.ndarray, 
                          target_id: str, 
                          output_dir: str = 'data/processed/pixel_diffs/') -> str:
    """
    Save pixel difference arrays to disk.
    
    Args:
        pixel_diffs: Array of pixel differences
        target_id: Target identifier
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{target_id}_pixel_diffs.npy")
    
    np.save(output_path, pixel_diffs)
    logger.info(f"Saved pixel differences to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Test the pixel processing pipeline
    logging.basicConfig(level=logging.INFO)
    
    # Test with a known Kepler target
    test_target = "757076"  # Kepler-10
    
    results = process_target_pixel_file(test_target)
    
    if results['success']:
        print("Pixel processing test successful!")
        print(f"Extracted {len([k for k in results.keys() if k.startswith('pixel_')])} pixel features")
        print(f"Extracted {len([k for k in results.keys() if k.startswith('crowdsap') or k.startswith('flux_frac') or k.startswith('background')])} crowding features")
        print(f"TPF shape: {results['tpf'].flux.shape}")
        print(f"Aperture size: {np.sum(results['aperture_mask'])} pixels")
    else:
        print("Pixel processing test failed!")