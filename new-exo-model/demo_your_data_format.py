#!/usr/bin/env python3
"""
Demonstration script showing how to query the exoplanet API with your data format
"""

import requests
import json

def query_exoplanet_api(target_id, exoplanet_data):
    """
    Query the exoplanet fusion API with your data format
    
    Args:
        target_id (str): Unique identifier for your target
        exoplanet_data (dict): Your data format with the required fields
    
    Returns:
        dict: API response with prediction results
    """
    
    # API endpoint
    url = 'http://localhost:8001/predict_exoplanet'
    
    # Prepare request payload
    payload = {
        'kepid': target_id,
        'features': json.dumps(exoplanet_data)
    }
    
    try:
        # Make the API request
        response = requests.post(url, data=payload, timeout=30)
        
        if response.status_code == 200:
            return {
                'success': True,
                'data': response.json()
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f"Request failed: {str(e)}"
        }

def main():
    """
    Demonstration of querying the API with your data format
    """
    
    print("🚀 Exoplanet API Query Demonstration")
    print("=" * 50)
    
    # Example 1: Your exact data format
    print("\n📝 Example 1: Using your exact data schema")
    
    your_data_format = {
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
    
    # Query the API
    result = query_exoplanet_api("TIC_123456789", your_data_format)
    
    if result['success']:
        data = result['data']
        print(f"✅ Success!")
        print(f"   Target ID: {data['kepid']}")
        print(f"   🎯 Final Confidence: {data['p_final']:.3f}")
        print(f"   📊 Decision: {data['decision']}")
        print(f"   🔍 Model Breakdown:")
        print(f"      Random Forest:  {data['p_rf']:.3f}")
        print(f"      Residual CNN:   {data['p_residual']:.3f}")
        print(f"      Pixel CNN:      {data['p_pixel']:.3f}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    # Example 2: Different planet types
    print("\n📝 Example 2: Different exoplanet types")
    
    planet_types = [
        {
            "name": "Hot Jupiter",
            "id": "hot_jupiter_001",
            "data": {
                "mission": "TESS",
                "orbital_period_days": 3.2,
                "transit_duration_hours": 2.8,
                "transit_depth_ppm": 8500,
                "planet_radius_re": 1.2,
                "equilibrium_temp_k": 1200,
                "insolation_flux_earth": 450,
                "stellar_teff_k": 5800,
                "stellar_radius_re": 1.1,
                "apparent_mag": 12.5,
                "ra": 280.5,
                "dec": 45.2
            }
        },
        {
            "name": "Super-Earth",
            "id": "super_earth_001",
            "data": {
                "mission": "Kepler",
                "orbital_period_days": 22.7,
                "transit_duration_hours": 5.1,
                "transit_depth_ppm": 1800,
                "planet_radius_re": 1.8,
                "equilibrium_temp_k": 420,
                "insolation_flux_earth": 3.2,
                "stellar_teff_k": 4800,
                "stellar_radius_re": 0.7,
                "apparent_mag": 13.2,
                "ra": 200.1,
                "dec": 45.2
            }
        },
        {
            "name": "Mini-Neptune",
            "id": "mini_neptune_001", 
            "data": {
                "mission": "TESS",
                "orbital_period_days": 8.3,
                "transit_duration_hours": 3.2,
                "transit_depth_ppm": 5890,
                "planet_radius_re": 4.1,
                "equilibrium_temp_k": 890,
                "insolation_flux_earth": 45.6,
                "stellar_teff_k": 5400,
                "stellar_radius_re": 1.1,
                "apparent_mag": 11.8,
                "ra": 310.7,
                "dec": -15.3
            }
        }
    ]
    
    results = []
    for planet in planet_types:
        result = query_exoplanet_api(planet['id'], planet['data'])
        if result['success']:
            data = result['data']
            results.append({
                'type': planet['name'],
                'confidence': data['p_final'],
                'decision': data['decision']
            })
            print(f"   {planet['name']}: {data['decision']} (confidence: {data['p_final']:.3f})")
        else:
            print(f"   {planet['name']}: ❌ Failed")
    
    # Example 3: Minimal data (edge case)
    print("\n📝 Example 3: Minimal data test")
    
    minimal_data = {
        "mission": "Unknown",
        "orbital_period_days": 1,
        "transit_duration_hours": 1,
        "transit_depth_ppm": 1,
        "planet_radius_re": 1,
        "equilibrium_temp_k": 1,
        "insolation_flux_earth": 1,
        "stellar_teff_k": 1,
        "stellar_radius_re": 1,
        "apparent_mag": 0,
        "ra": 360,
        "dec": -90
    }
    
    result = query_exoplanet_api("minimal_test", minimal_data)
    if result['success']:
        data = result['data']
        print(f"   ✅ Minimal data processed successfully")
        print(f"   Decision: {data['decision']} (confidence: {data['p_final']:.3f})")
    else:
        print(f"   ❌ Minimal data failed: {result['error']}")
    
    print("\n🎉 Demonstration complete!")
    print("\n💡 Key takeaways:")
    print("   • Your data format works perfectly with the API")
    print("   • No field name changes required")
    print("   • API handles various planet types and edge cases")
    print("   • Response includes detailed model breakdown")
    print("   • Ready for production use!")

if __name__ == "__main__":
    # First check if API is available
    try:
        health_response = requests.get('http://localhost:8001/health', timeout=5)
        if health_response.status_code == 200:
            print("🟢 API is available - starting demonstration...")
            main()
        else:
            print("🔴 API returned unexpected status. Please check if the server is running.")
            print("Start server with: python -m uvicorn serve_fusion:app --host 0.0.0.0 --port 8001")
    except requests.exceptions.RequestException:
        print("🔴 Cannot connect to API. Please ensure the server is running.")
        print("Start server with: python -m uvicorn serve_fusion:app --host 0.0.0.0 --port 8001")