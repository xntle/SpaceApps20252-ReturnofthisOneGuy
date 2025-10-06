# 🚀 API Query Guide - Using Your Data Format

## 📋 **Your Data Schema**

```json
{
  "mission": "string",
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
```

---

## 🔧 **How to Query the API**

### **Base URL**: `http://localhost:8001`

### **Endpoint**: `POST /predict_exoplanet`

---

## 💻 **Method 1: Python with requests**

```python
import requests
import json

# Your exoplanet data
exoplanet_data = {
    "mission": "TESS",                    # Mission name (Kepler/TESS/etc)
    "orbital_period_days": 15.73,         # Orbital period in days
    "transit_duration_hours": 4.2,       # Transit duration in hours
    "transit_depth_ppm": 3200,           # Transit depth in parts per million
    "planet_radius_re": 2.1,             # Planet radius in Earth radii
    "equilibrium_temp_k": 680,           # Equilibrium temperature in Kelvin
    "insolation_flux_earth": 8.5,        # Insolation flux relative to Earth
    "stellar_teff_k": 5200,              # Stellar effective temperature in K
    "stellar_radius_re": 0.9,            # Stellar radius in Solar radii
    "apparent_mag": 13.1,                # Apparent magnitude
    "ra": 200.5,                         # Right ascension in degrees
    "dec": 35.7                          # Declination in degrees
}

# Make the API request
response = requests.post(
    'http://localhost:8001/predict_exoplanet',
    data={
        'kepid': 'TIC_123456789',         # Your target identifier
        'features': json.dumps(exoplanet_data)
    }
)

# Process the response
if response.status_code == 200:
    result = response.json()
    print(f"🎯 Exoplanet Analysis Results:")
    print(f"   Target ID: {result['kepid']}")
    print(f"   🤖 AI Confidence: {result['p_final']:.3f}")
    print(f"   📊 Decision: {result['decision']}")
    print(f"   🔍 Model Breakdown:")
    print(f"      Random Forest:  {result['p_rf']:.3f}")
    print(f"      Residual CNN:   {result['p_residual']:.3f}")
    print(f"      Pixel CNN:      {result['p_pixel']:.3f}")
else:
    print(f"❌ Error: {response.status_code}")
    print(f"Response: {response.text}")
```

### **Expected Output:**
```
🎯 Exoplanet Analysis Results:
   Target ID: TIC_123456789
   🤖 AI Confidence: 0.678
   📊 Decision: CONFIRMED
   🔍 Model Breakdown:
      Random Forest:  0.720
      Residual CNN:   0.650
      Pixel CNN:      0.665
```

---

## 🌐 **Method 2: cURL Command**

```bash
curl -X POST "http://localhost:8001/predict_exoplanet" \
  -H "Content-Type: multipart/form-data" \
  -F 'kepid=TIC_123456789' \
  -F 'features={
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
  }'
```

### **Expected Response:**
```json
{
  "kepid": "TIC_123456789",
  "p_rf": 0.720,
  "p_residual": 0.650,
  "p_pixel": 0.665,
  "p_final": 0.678,
  "decision": "CONFIRMED"
}
```

---

## 📊 **Method 3: Batch Processing**

```python
import requests
import json

# Multiple exoplanet candidates
candidates = [
    {
        "id": "Hot_Jupiter_001",
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
        "id": "Super_Earth_002", 
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
    }
]

# Process each candidate
results = []
for candidate in candidates:
    response = requests.post(
        'http://localhost:8001/predict_exoplanet',
        data={
            'kepid': candidate['id'],
            'features': json.dumps(candidate['data'])
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        results.append({
            'target': candidate['id'],
            'confidence': result['p_final'],
            'decision': result['decision']
        })
        print(f"✅ {candidate['id']}: {result['decision']} ({result['p_final']:.3f})")
    else:
        print(f"❌ {candidate['id']}: Failed")

# Summary
confirmed = len([r for r in results if r['decision'] == 'CONFIRMED'])
print(f"\n📊 Batch Results: {confirmed}/{len(results)} candidates confirmed")
```

---

## 🔍 **Method 4: Health Check First**

```python
import requests

# Always check API health before making predictions
def check_api_health():
    try:
        response = requests.get('http://localhost:8001/health')
        if response.status_code == 200:
            health = response.json()
            print("🟢 API Status: HEALTHY")
            print(f"   Models Loaded: {health['models_loaded']}")
            print(f"   Device: {health['device']}")
            print(f"   Threshold: {health['threshold']}")
            return True
        else:
            print(f"🔴 API Status: UNHEALTHY ({response.status_code})")
            return False
    except Exception as e:
        print(f"🔴 API Status: UNREACHABLE ({e})")
        return False

# Check health before proceeding
if check_api_health():
    # Your prediction code here
    exoplanet_data = {
        "mission": "TESS",
        "orbital_period_days": 15.73,
        # ... rest of your data
    }
    # Make prediction...
else:
    print("❌ Cannot proceed - API not available")
```

---

## 🌟 **Method 5: Real-world Example**

```python
import requests
import json

def analyze_exoplanet(target_id, exoplanet_params):
    """
    Analyze an exoplanet candidate using the fusion AI model
    
    Args:
        target_id (str): Unique identifier for the target
        exoplanet_params (dict): Your data format with all parameters
    
    Returns:
        dict: Analysis results with confidence and decision
    """
    
    # API endpoint
    url = 'http://localhost:8001/predict_exoplanet'
    
    # Prepare the request
    payload = {
        'kepid': target_id,
        'features': json.dumps(exoplanet_params)
    }
    
    try:
        # Make the request
        response = requests.post(url, data=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Interpret the results
            confidence = result['p_final']
            decision = result['decision']
            
            # Add interpretation
            if confidence >= 0.7:
                interpretation = "HIGH confidence exoplanet"
            elif confidence >= 0.5:
                interpretation = "MODERATE confidence exoplanet"
            elif confidence >= 0.3:
                interpretation = "LOW confidence exoplanet"
            else:
                interpretation = "Likely FALSE POSITIVE"
            
            return {
                'target_id': target_id,
                'confidence': confidence,
                'decision': decision,
                'interpretation': interpretation,
                'model_scores': {
                    'random_forest': result['p_rf'],
                    'residual_cnn': result['p_residual'],
                    'pixel_cnn': result['p_pixel']
                },
                'status': 'success'
            }
        else:
            return {
                'target_id': target_id,
                'status': 'error',
                'error': f"HTTP {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        return {
            'target_id': target_id,
            'status': 'error',
            'error': str(e)
        }

# Example usage
candidate_data = {
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

# Analyze the candidate
result = analyze_exoplanet("TIC_123456789", candidate_data)

if result['status'] == 'success':
    print(f"🎯 Analysis Complete for {result['target_id']}")
    print(f"   🤖 AI Confidence: {result['confidence']:.3f}")
    print(f"   📊 Decision: {result['decision']}")
    print(f"   💡 Interpretation: {result['interpretation']}")
else:
    print(f"❌ Analysis failed: {result['error']}")
```

---

## 📋 **Response Format**

### **Success Response (200 OK):**
```json
{
  "kepid": "your_target_id",
  "p_rf": 0.720,           // Random Forest probability
  "p_residual": 0.650,     // Residual CNN probability  
  "p_pixel": 0.665,        // Pixel CNN probability
  "p_final": 0.678,        // Final ensemble probability
  "decision": "CONFIRMED"   // CONFIRMED or FALSE_POSITIVE
}
```

### **Error Response (4xx/5xx):**
```json
{
  "detail": "Error description"
}
```

---

## 🔧 **Field Mapping & Units**

| Your Field | Expected Unit | Description |
|------------|---------------|-------------|
| `mission` | string | Mission name (info only) |
| `orbital_period_days` | days | Orbital period |
| `transit_duration_hours` | hours | Duration of transit |
| `transit_depth_ppm` | ppm | Transit depth (parts per million) |
| `planet_radius_re` | R⊕ | Planet radius in Earth radii |
| `equilibrium_temp_k` | Kelvin | Planet equilibrium temperature |
| `insolation_flux_earth` | relative | Insolation flux relative to Earth |
| `stellar_teff_k` | Kelvin | Stellar effective temperature |
| `stellar_radius_re` | R☉ | Stellar radius in Solar radii |
| `apparent_mag` | magnitude | Apparent magnitude of star |
| `ra` | degrees | Right ascension (0-360) |
| `dec` | degrees | Declination (-90 to +90) |

---

## 🚀 **Quick Start Template**

```python
import requests
import json

# 1. Your data
data = {
    "mission": "YOUR_MISSION",
    "orbital_period_days": YOUR_PERIOD,
    "transit_duration_hours": YOUR_DURATION,
    "transit_depth_ppm": YOUR_DEPTH,
    "planet_radius_re": YOUR_RADIUS,
    "equilibrium_temp_k": YOUR_TEMP,
    "insolation_flux_earth": YOUR_FLUX,
    "stellar_teff_k": YOUR_STELLAR_TEMP,
    "stellar_radius_re": YOUR_STELLAR_RADIUS,
    "apparent_mag": YOUR_MAG,
    "ra": YOUR_RA,
    "dec": YOUR_DEC
}

# 2. Make request
response = requests.post(
    'http://localhost:8001/predict_exoplanet',
    data={'kepid': 'YOUR_ID', 'features': json.dumps(data)}
)

# 3. Get result
if response.status_code == 200:
    result = response.json()
    print(f"Confidence: {result['p_final']:.3f}")
    print(f"Decision: {result['decision']}")
else:
    print(f"Error: {response.text}")
```

---

## ✅ **Your data format is 100% compatible - ready to use!**

Just replace the example values with your actual data and start querying! 🚀