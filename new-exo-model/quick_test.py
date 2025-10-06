#!/usr/bin/env python3
"""
Quick one-liner test for your data format
"""

import requests
import json

# Your data format - just replace with your actual values
data = {
    "mission": "TESS",
    "orbital_period_days": 15.73,
    "transit_duration_hours": 4.2,
    "transit_depth_ppm": 3200,
    "planet_radius_re": 2.1,
    "equilibrium_temp_k": 680,
    "insolation_flux_earth": 8.5,
    "stellar_teff_k": 5200,
    "stellar_radius_re": 0.9,
    "apparent_mag": 13.1,
    "ra": 200.5,
    "dec": 35.7
}

# One-liner API query
result = requests.post('http://localhost:8001/predict_exoplanet', data={'kepid': 'test', 'features': json.dumps(data)}).json()

# Print results
print(f"🎯 Confidence: {result['p_final']:.3f}")
print(f"📊 Decision: {result['decision']}")
print(f"🔍 Breakdown: RF={result['p_rf']:.3f}, CNN1={result['p_residual']:.3f}, CNN2={result['p_pixel']:.3f}")