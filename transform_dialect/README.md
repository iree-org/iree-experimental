# Transform Dialect Tooling

This directory contains a few simple tools to enhance productivity of IREE
experiments based on the transform dialect.

## Script

Contains shell functions that simplify running IREE tools with the transform
dialect. These should generally be considered brittle but may be useful to some.

First, source the script file:
```
source iree-samples/transform_dialect/scripts.sh
```

For execution, assumptions are:
  1. execution starts from the IREE source dir root
  2. the IREE build directory lives at `./build`
  3. `mlir-opt` is in the path
  4. `./build/tools` (i.e. `iree-opt`, `iree-compile`, `iree-run-module`) is in the path
  5. for CUDA execution, `nvprof` is in the path.

Disclaimers:
  1. since the exploration process can involve modifying source IR, transform IR
  and the compiler itself, each command compiles what it needs to simplify and
  automate the process. This may be surprising if one does not want recompilation
  to occur.
  2. examples are restricted to the CUDA case for now.
  3. examples are restricted to 2-D reduction cases for now.
  4. the stub file can have multiple functions, but they all should have private
  visibility. We use a poor man's `sed`-based filter of the function we care
  about by replacing the visibility of that one function and letting
  `mlir-opt -symbol-dce` clean the IR for us.

Note: the underlying helper functions `iree-transform-xxx` can be parameterized to
other backends and CUDA is not a strong restriction, just a convenience.

### Create a New Problem To Map

Run:
```
(\
  benchmark-transform-create \
  -b cuda \
  <iree-samples-dir-path>/transform_dialect/benchmark_linalg_reductions.stub.mlir \
  reduction_2d_static \
  f32 \
  123 \
  456 \
)
```

This should print something resembling:
```
==========================================================
Problem created successfully, reproduction instructions:
==========================================================
Transform dialect source file is: /tmp/reduction_2d_static_123x456.mlir
Transform dialect transform file is: /tmp/iree_transform_dialect_ac2e60.mlir
Dump transformed IR with: benchmark-transform-run-iree-opt -b cuda /tmp/reduction_2d_static_123x456.mlir /tmp/iree_transform_dialect_ac2e60.mlir
Dump transformed PTX with: benchmark-transform-run-iree-compile -b cuda /tmp/reduction_2d_static_123x456.mlir /tmp/iree_transform_dialect_ac2e60.mlir
Run nvprof with e.g.: benchmark-transform-run-nvprof /tmp/reduction_2d_static_123x456.mlir /tmp/iree_transform_dialect_ac2e60.mlir reduction_2d_static 123 456
==========================================================
```


### Produce Transformed IR

Manually modify the content of the transform IR file (or not) (i.e. /tmp/iree_transform_dialect_ac2e60.mlir), then run:

```
( \
  benchmark-transform-run-iree-opt \
  -b cuda \
  /tmp/reduction_2d_static_123x456.mlir \
  /tmp/iree_transform_dialect_ac2e60.mlir \
)
```

This should print the transformed IR (or appropriate error messages when relevant).

This provides an easy to interact with interpreted mode.

### Produce Transformed PTX

Once the transformed IR is in a satisfactory state, one can inspect the PTX.

```
( \
  benchmark-transform-run-iree-compile \
  -b cuda \
  /tmp/reduction_2d_static_123x456.mlir \
  /tmp/iree_transform_dialect_ac2e60.mlir \
)
```

This should print the transformed PTX (or appropriate error messages when relevant).

## Run and Benchmark

The following requires nvprof to be in the PATH.

When things look good, one can execute, get a filtered nvprof trace and a rough
estimate of the performance by pasting the last repro command:

```
(
  benchmark-transform-run-nvprof \
  -b cuda \
  /tmp/reduction_2d_static_123x456.mlir \
  /tmp/iree_transform_dialect_ac2e60.mlir \
  reduction_2d_static \
  f32 \
  123 \
  456 \
)
```

Prints something resembling:
```
==========================================================
Reproduction instructions:
iree-transform-compile /tmp/reduction_2d_static_123x456.mlir -b cuda -c /tmp/iree_transform_dialect_ac2e60.mlir -- --iree-hal-benchmark-dispatch-repeat-count=6 | \
nvprof --print-gpu-trace iree-run-module --entry_function=reduction_2d_static --device=cuda --function_input="123x456xf32=1" | \
grep reduction_2d_static
==========================================================
==3499964== NVPROF is profiling process 3499964, command: iree-run-module --entry_function=reduction_2d_static --device=cuda --function_input="123x456xf32=1"
EXEC @reduction_2d_static
==3499964== Profiling application: iree-run-module --entry_function=reduction_2d_static --device=cuda --function_input="123x456xf32=1"
388.21ms  596.26us            (123 1 1)       (128 1 1)        18        0B      768B  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x456 [24]
388.80ms  3.2960us            (123 1 1)       (128 1 1)        18        0B      768B  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x456 [25]
388.81ms  2.8800us            (123 1 1)       (128 1 1)        18        0B      768B  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x456 [26]
388.81ms  2.9130us            (123 1 1)       (128 1 1)        18        0B      768B  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x456 [27]
388.82ms  2.8800us            (123 1 1)       (128 1 1)        18        0B      768B  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x456 [28]
388.82ms  2.8800us            (123 1 1)       (128 1 1)        18        0B      768B  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x456 [29]
reduction_2d_static --function_input="123x456xf32=1" P50: 2880.0000 ns 19.47500000000000000000 GElements/s
```

Note how the first instance takes significantly longer due to initial host->device
memory copies.

The reproduction instructions can be run independently and adapted.

It is the responsibility of the user to convert the number of elements processed
per second (i.e. the product of the sizes) to the relevant metric for the benchmark.
In the case of reduction_2d_static, a `tensor<123x456xf32>` is read and reduced
into a `tensor<123xf32>`.
This corresponds to 4 bytes read per element and a negligible amount of bytes
written per element.

This gives us roughly `80GB/s` read bandwidth, in `2.8us` (latency-bound).

We can generate and run another problem by adding the `-r` argument:
```
( \
  benchmark-transform-create -r \
  -b cuda \
  <iree-samples-dir-path>/transform_dialect/benchmark_linalg_reductions.stub.mlir \
  reduction_2d_static \
  f32 \
  123 \
  45678 \
)
```

The various repro instructions are printed and the problem is also run:
```
==3504051== NVPROF is profiling process 3504051, command: iree-run-module --entry_function=reduction_2d_static --device=cuda --function_input="123x45678xf32=1"
EXEC @reduction_2d_static
==3504051== Profiling application: iree-run-module --entry_function=reduction_2d_static --device=cuda --function_input="123x45678xf32=1"
427.44ms  10.241ms            (123 1 1)       (256 1 1)        24        0B  1.2500KB  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x45678 [24]
437.69ms  42.434us            (123 1 1)       (256 1 1)        24        0B  1.2500KB  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x45678 [25]
437.73ms  42.499us            (123 1 1)       (256 1 1)        24        0B  1.2500KB  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x45678 [26]
437.77ms  42.498us            (123 1 1)       (256 1 1)        24        0B  1.2500KB  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x45678 [27]
437.82ms  42.531us            (123 1 1)       (256 1 1)        24        0B  1.2500KB  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x45678 [28]
437.86ms  42.530us            (123 1 1)       (256 1 1)        24        0B  1.2500KB  NVIDIA GeForce          1        13                     -                -  reduction_2d_static_dispatch_0_generic_123x45678 [29]
reduction_2d_static --function_input="123x45678xf32=1" P50: 42499.000 ns 132.20061648509376691216 GElements/s
```

This corresponds to roughly `528GB/s` read bandwidth (i.e. 4B / element with `f32`).

As a rough point of reference, running the CUDA samples
[bandwidth test](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/bandwidthTest) on this author's machine runs at roughly `520GB/s`.

The full dump is:

```
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: NVIDIA GeForce RTX 2080 Ti
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     7.8

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     9.0

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     519.5

Result = PASS
```

As a better point of reference, running the BabelStream [benchmarks](https://github.com/UoB-HPC/BabelStream)
on this author's machine runs at roughly `550GB/s`.

The full dump is:

```
BabelStream
Version: 4.0
Implementation: CUDA
Running kernels 100 times
Precision: double
Array size: 268.4 MB (=0.3 GB)
Total size: 805.3 MB (=0.8 GB)
Using CUDA device NVIDIA GeForce RTX 2080 Ti
Driver: 11060
Function    MBytes/sec  Min (sec)   Max         Average     
Copy        547796.718  0.00098     0.00101     0.00100
Mul         544875.681  0.00099     0.00101     0.00100
Add         556654.794  0.00145     0.00148     0.00147
Triad       556770.252  0.00145     0.00148     0.00147
Dot         574513.619  0.00093     0.00096     0.00094
```
