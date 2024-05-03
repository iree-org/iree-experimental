# IREE Profiling Tools

IREE uses [tracy](http://github.com/wolfpld/tracy) for profiling. See more
details about IREE profiling
[here](https://iree.dev/developers/performance/profiling/).

This directory contains various tools to augument tracy profiling, for example,
converting tracy files into chrome json files which can be loaded into
[perfetto UX](http://ui.perfetto.dev) or [chrome://tracing](chrome://tracing).

## Build

For Android build, see [Build for Android](#build-for-android) below.

IREE profiling tools assume IREE repository has already been cloned in the same
build machine. Once both iree and iree-experimental are cloned, IREE profiling
can be built with cmake. `../../iree-prof-build` is given as output directory
not to mix output files with source files.

```shell
cd iree-experimental/iree-prof-tools
cmake -G Ninja -B ../../iree-prof-build/ .
cmake --build ../../iree-prof-build/
```

By default, it assumes the directory of IREE root is `../../iree` from the
`iree-prof-tools` directory. It can be overridden by `IREE_ROOT_DIR` flag.

```shell
cmake -G Ninja -B ../../iree-prof-build/ -S . -DIREE_ROOT_DIR=foo/bar
```

## iree-prof

`iree-prof` is a tool to make it easier to run commands for profiling, e.g.
`iree-run-module` instead of `iree-tracy-capture`. It also prints some useful
profiling results in stdout.

```shell
../../iree-prof-build/iree-prof --output_tracy_file=/tmp/prof.tracy -- \
  ./../iree-build/tools/iree-run-module --device=local-task \
  --module=/tmp/mobilenet_v2.vmfb --function=predict \
  --input="1x224x224x3xf32=0"
```

Output looks like:

```
<output from iree-run-module>
[TRACY    ] Capture Name: iree-run-module @ <date> <time>
[TRACY    ]     Cpu Arch: x86_64
[TRACY    ]
[TRACY-CPU]  CPU Threads: 9
[TRACY-CPU]    CPU Zones: 19575
...
```

## iree-prof-convert

`iree-prof-convert` is a tool to convert a tracy file into a chrome json file
and/or a comma-separate-values (csv) file.

```shell
../../iree-prof-build/iree-prof-convert --input_tracy_file=/tmp/prof.tracy \
  --output_chrome_file=/tmp/prof.json --output_csv_file=/tmp/prof.csv
```

The output json file can be loaded into
[perfetto UX](http://ui.perfetto.dev) or [chrome://tracing](chrome://tracing).
Note that the csv file output by `iree-prof-convert` contains the same
information in stdout, which is different from one by `iree-tracy-csvexport`.

## Build for Android

Once IREE is built for Android as explained
[here](https://iree.dev/building-from-source/android/), IREE profiling tools
can be built for Android with 2 more options,
`-DIREE_PROF_BUILD_TRACY_DEPS=ON` and `-DCMAKE_CXX_FLAGS="-DNO_PARALLEL_SORT"`.

```shell
cd iree-experimental/iree-prof-tools
cmake -G Ninja -B ../../iree-prof-build-android/ \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DIREE_HOST_BIN_DIR="$PWD/../../iree-build/install/bin" \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM="android-29" \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_PROF_BUILD_TRACY_DEPS=ON \
  -DCMAKE_CXX_FLAGS="-DNO_PARALLEL_SORT" \
  .
cmake --build ../../iree-prof-build-android/
```

On latest arm64 Android devices,
[memory tagged extension (MTE)](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/enhancing-memory-safety)
is enabled by default while Tracy packs pointers to reduce space and cause
`iree-prof` and `iree-prof-convert` to crash. To disable it globally, set
`/proc/sys/abi/tagged_addr_disabled` to `0` with `adb`.

```shell
adb root
adb shell "echo 1 > /proc/sys/abi/tagged_addr_disabled"
```
