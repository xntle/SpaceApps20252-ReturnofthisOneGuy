# 🚀 CNN Coverage Expansion Strategy

## Current Status
- **Current Coverage**: 0.7% (72 residual windows, 58 pixel differences)
- **Target Coverage**: >5% for effective fusion
- **Need**: ~500+ successful downloads for each data type

## 🎯 Strategies to Increase CNN Coverage

### 1. **Smart Target Filtering** ⭐ (Currently Running)
```python
# High-success targets
df_filtered = df[
    (df['koi_period'] > 0.5) & (df['koi_period'] < 100) &    # Valid periods
    (df['koi_duration'] > 0.5) & (df['koi_duration'] < 24) &  # Valid durations  
    (df['koi_depth'] > 0) &                                   # Must have depth
    (pd.notna(df['koi_period'])) &                           # No NaN values
    (pd.notna(df['koi_time0bk']))
]
```

### 2. **Parallel Processing** ⚡
```python
# Use ThreadPoolExecutor for concurrent downloads
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_target, args) for args in target_args]
```

### 3. **Retry Logic & Error Handling** 🔄
```python
def robust_download_with_retry(kepid, max_retries=3):
    for attempt in range(max_retries):
        try:
            lc = download_lightcurve(str(kepid), mission='Kepler')
            if lc is not None:
                return lc
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    return None
```

### 4. **Cache Cleaning** 🧹
```bash
# Remove corrupted cache files
rm -rf ~/.lightkurve/cache/mastDownload/
```

### 5. **Target Prioritization** 📊
- **Confirmed planets first** (higher success rate)
- **Short periods (1-10 days)** (more transits available)
- **Bright stars** (better data quality)
- **Multiple quarters available**

### 6. **Batch Processing Strategy** 📦
```python
# Process in batches of 100-200 targets
batches = [df_targets[i:i+100] for i in range(0, len(df_targets), 100)]
for batch in batches:
    process_batch(batch)
    time.sleep(60)  # Rate limiting
```

### 7. **Alternative Data Sources** 🌟
- **TESS data**: `mission='TESS'` for newer targets
- **K2 data**: `mission='K2'` for extended mission
- **Pre-computed features**: Use existing light curve databases

## 🔧 Immediate Actions

### Run These Commands:
```bash
# 1. Clean corrupted cache
rm -rf ~/.lightkurve/cache/mastDownload/Kepler/kplr010666592*
rm -rf ~/.lightkurve/cache/mastDownload/Kepler/kplr012366084*

# 2. Run optimized expansion (currently running)
cd scripts && python optimized_cnn_expansion.py

# 3. Monitor progress
watch -n 30 'ls ../data/processed/residual_windows/*.npy | wc -l'
```

### Quick Coverage Boost:
```python
# Focus on these high-success targets:
confirmed_planets = df[
    (df['koi_disposition'] == 'CONFIRMED') &
    (df['koi_period'] > 1) & (df['koi_period'] < 10) &
    (df['koi_duration'] > 1) & (df['koi_duration'] < 6) &
    (df['koi_depth'] > 100)  # Deep transits
].head(200)
```

## 📈 Expected Results

### Coverage Targets:
- **Conservative**: 3-5% (300-500 files each)
- **Optimistic**: 8-10% (800-1000 files each)  
- **Maximum**: 15-20% (1500-2000 files each)

### Success Rates by Target Type:
- **Confirmed planets**: 60-80% success
- **False positives**: 40-60% success
- **Short periods**: 70-90% success
- **Long periods**: 30-50% success

## 🏆 Fusion Performance Expectations

### At Different Coverage Levels:
- **1% coverage**: 96.5% AUC (current)
- **5% coverage**: 97-98% AUC (estimated)
- **10% coverage**: 98-99% AUC (target)
- **20% coverage**: 99%+ AUC (optimal)

## ⚠️ Common Issues & Solutions

### 1. **Corrupted Downloads**
```bash
# Solution: Clean cache and retry
rm -rf ~/.lightkurve/cache/mastDownload/Kepler/kplr{target_id}*
```

### 2. **Rate Limiting**
```python
# Solution: Add delays between requests
time.sleep(1)  # 1 second between downloads
```

### 3. **Memory Issues**
```python
# Solution: Process in smaller batches
batch_size = 50  # Reduce from 500
```

### 4. **Network Timeouts**
```python
# Solution: Increase timeout and add retries
lk.search_lightcurve(target, timeout=60)
```

## 🚀 Next Steps

1. **Monitor current expansion** (optimized_cnn_expansion.py)
2. **Clean corrupted cache files**
3. **Run standardization** after expansion
4. **Retrain enhanced fusion model**
5. **Compare against 99.41% tabular baseline**

The key is **persistence** - even with 50% success rate, processing 1000 targets gives 500 files (5% coverage)!