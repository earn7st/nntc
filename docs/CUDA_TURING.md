# Turing encoder support

The CUDA encoder supports compute capability 7.5 and newer when the build contains a compatible kernel image. Runtime probing checks both the capability floor and kernel-image compatibility. Auto mode retains CPU fallback; explicit cuda mode reports errors.

Build on a CUDA 13.4 / Visual Studio 2022 workstation:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -T cuda=13.4 -DCMAKE_CUDA_ARCHITECTURES=75 -DNNTC_CUDA_ARCHITECTURES=75
cmake --build build --config Release --target nntc_encode -j 4
```

NNTC_CUDA_ARCHITECTURES is a cache option. It inherits CMAKE_CUDA_ARCHITECTURES on first configuration; set it explicitly to change an existing build. The default architecture list includes 75. Existing CUDA toolkit minimum requirements remain unchanged.

Validated on 2026-11-23 with RTX 2080 SUPER (7.5), driver 591.86, CUDA toolkit 13.4:

- tests/tiny.png, c0=2 c1=4, rounds=2, backend check: zero mismatches and flips.
- Same tiny input, backend auto: selected CUDA and completed.
- Same tiny input, c0=4 c1=4, rounds=2, explicit CUDA: completed.
- Logs and generated outputs: build/cuda75-validation/.

These are smoke and kernel-comparison checks, not the complete release gate or a full-material performance benchmark. Numerical encoding kernels and fast-math settings were not changed. No Unity editor validation was performed.
