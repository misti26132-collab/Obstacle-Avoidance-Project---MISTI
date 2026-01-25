#!/usr/bin/env python3
import torch
import time

print("=" * 60)
print("GPU DIAGNOSTIC")
print("=" * 60)

# Check CUDA
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    
    # Test GPU performance
    print("\n2. Testing GPU Performance...")
    
    # Create large tensor on GPU
    x = torch.randn(5000, 5000).cuda()
    
    # Time matrix multiplication on GPU
    start = time.time()
    for _ in range(10):
        y = torch.matmul(x, x)
    gpu_time = time.time() - start
    
    print(f"   GPU Time: {gpu_time:.3f}s")
    
    # Compare with CPU
    x_cpu = x.cpu()
    start = time.time()
    for _ in range(10):
        y_cpu = torch.matmul(x_cpu, x_cpu)
    cpu_time = time.time() - start
    
    print(f"   CPU Time: {cpu_time:.3f}s")
    print(f"   Speedup: {cpu_time/gpu_time:.1f}x faster on GPU")
    
    # Memory info
    print(f"\n3. GPU Memory:")
    print(f"   Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"   Total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
else:
    print("\n❌ CUDA NOT AVAILABLE!")
    print("   Possible causes:")
    print("   - Wrong PyTorch version (need Jetson-specific)")
    print("   - CUDA not installed")
    print("   - Driver issues")

print("=" * 60)