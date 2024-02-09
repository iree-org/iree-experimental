# IREE Profiling Tools

IREE uses [tracy](http://github.com/wolfpld/tracy) for profiling. See more
details about IREE profiling
[here](https://iree.dev/developers/performance/profiling/).

This directory contains various tools to augument tracy profiling, for example,
converting tracy files into chrome json files which can be loaded into
[perfetto UX](http://ui.perfetto.dev) or [chrome://tracing](chrome://tracing).

## Build

IREE profiling tools assume IREE repository has already been cloned in the same
build machine. Once both iree and iree-samples are cloned, IREE profiling can be
built with cmake. `../../iree-prof-build` is given as output directory not to
mix output files with source files.

```shell
cd iree-samples/iree-prof-tools
cmake -G Ninja -B ../../iree-prof-build/ -S .
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
[TRACY    ] Capture Name: iree-run-module @ <date>
[TRACY    ]     Cpu Arch: x86_64
[TRACY    ]
[TRACY-CPU]  CPU Threads: 9
[TRACY-CPU]    CPU Zones: 19575
...
```

## iree-prof-convert

`iree-prof-convert` is a tool to convert a tracy file into a chrome json file.

```shell
../../iree-prof-build/iree-prof-convert --input_tracy_file=/tmp/prof.tracy \
  --output_chrome_file=/tmp/prof.json
```

The output json file can be loaded into
[perfetto UX](http://ui.perfetto.dev) or [chrome://tracing](chrome://tracing).
