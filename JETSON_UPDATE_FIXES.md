# Jetson Orin Nano Update - Fixes Applied

## Issues Fixed

### 1. **RAM Memory Issue**
The recent Jetson update appears to have stricter memory management. The code now includes aggressive garbage collection to prevent out-of-memory errors.

**Changes made:**
- Added `AGGRESSIVE_MEMORY_CLEANUP` config option (enabled by default)
- Added `CUDA_MEMORY_THRESHOLD` to monitor and clear cache when >85% full
- Added `GC_COLLECT_INTERVAL` for frequent garbage collection (every 30 frames)
- Added `CUDA_EMPTY_CACHE_INTERVAL` for CUDA memory cleanup (every 50 frames)
- Implemented `cleanup_memory()` and `check_cuda_memory()` functions in main.py
- Memory cleanup now happens more frequently during inference
- Added `psutil` import for potential future RAM monitoring

**How it works:**
- Every 30 frames, the system runs `gc.collect()` and `torch.cuda.empty_cache()`
- Monitors CUDA memory usage and clears cache if it exceeds 85% threshold
- Deletes intermediate tensors after depth estimation to free memory immediately

### 2. **IP/Network Problem** (Model Download Issues)
The code previously had no fallback when network was unavailable or slow, causing the application to hang when trying to download MiDaS models.

**Changes made:**
- Added `MODEL_CACHE_DIR` config for local model caching
- Added `MAX_MODEL_LOAD_RETRIES` for resilient model loading
- Added `MODEL_LOAD_TIMEOUT` to prevent indefinite waiting (default: 60 seconds)
- Added `OFFLINE_MODE` option to use only cached models
- Implemented socket timeout for torch.hub operations
- Added fallback transforms if network load fails
- Added local cache fallback if network is unavailable
- Better error messages for debugging

**How it works:**
1. Attempts to load MiDaS from network with 60-second timeout
2. If network fails, checks local cache directory (`.cache/models/`)
3. If local cache exists, uses it immediately
4. If neither network nor cache available and `OFFLINE_MODE=True`, raises clear error
5. Otherwise retries network up to 3 times with exponential backoff

### 3. **YOLO Model Memory Optimization**
YOLO models are now loaded with explicit FP16 (half-precision) mode for Jetson.

**Changes made:**
- Models are now explicitly set to `.eval()` mode
- FP16 conversion applied for inference optimization
- Reduced memory footprint of YOLO models on CUDA device

## Configuration Options

Add these to your `config.py` to customize behavior:

```python
# Memory Management
AGGRESSIVE_MEMORY_CLEANUP = True      # Enable aggressive cleanup (default: True)
CUDA_MEMORY_THRESHOLD = 0.85          # Clear cache when >85% full
GC_COLLECT_INTERVAL = 30              # Run gc.collect() every N frames
CUDA_EMPTY_CACHE_INTERVAL = 50        # Clear CUDA cache every N frames

# Model Loading & Network Resilience
MODEL_CACHE_DIR = ".cache/models"     # Where to cache downloaded models
MAX_MODEL_LOAD_RETRIES = 5            # Number of retry attempts
MODEL_LOAD_TIMEOUT = 60               # Timeout in seconds for downloads
OFFLINE_MODE = False                  # Use only cached models if True
SKIP_MIDAS_DOWNLOAD = False           # Skip MiDaS if using local copy
```

## Troubleshooting

### If you get "RAM full" errors:
1. Increase `GC_COLLECT_INTERVAL` (default 30) to lower value like 20
2. Increase `CUDA_MEMORY_THRESHOLD` (default 0.85) to lower value like 0.75
3. Reduce `YOLO_IMG_SIZE` in config.py (try 256 instead of 288)

### If models fail to download:
1. Check internet connection
2. Try manually running this command to pre-cache the model:
   ```bash
   python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)"
   ```
3. Set `OFFLINE_MODE = True` in config.py to use only cached models
4. Increase `MODEL_LOAD_TIMEOUT` (default 60 seconds) if network is slow

### If the system hangs during startup:
- This is likely a network timeout. The code now has a 60-second timeout
- Check your network connectivity
- Try setting `OFFLINE_MODE = True` if models are already cached

## Testing the Fixes

Run the application as normal:
```bash
python main.py
```

Monitor memory usage (in another terminal):
```bash
watch -n 1 nvidia-smi
```

You should see CUDA memory being cleared periodically, and the application should be more stable with the Jetson update.

## Performance Notes

- FP16 mode significantly reduces memory usage (~50% less VRAM)
- Aggressive garbage collection may cause minor FPS dips (usually imperceptible)
- First run may take longer if models need to be downloaded and cached
- Subsequent runs will use cached models and startup will be much faster

## Additional Improvements

- Added logging for memory status and network operations
- Fallback transforms if network unavailable during startup
- Better error messages for network issues
- Cleaner memory handling after inference operations
