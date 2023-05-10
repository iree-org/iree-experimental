# IREE runtime library builder

Integrators often like to link against a single runtime library, and the
method of doing so is naturally owned by the integrator -- not necessarily
by IREE itself (i.e. IREE will never release a full libireert.a). This
separation of concerns is of practical importance because IREE's low level
runtime API is fine-grained and geared towards usage via LTO style
optimizations. This makes it inherently non-distributable, since every
toolchain defines such features and interop differently.

Also, it is often convenient during early development (of language bindings,
etc) to simply dynamically link to something that works, even if not optimal.
Because it cannot produce the best integration, IREE itself does not export
a shared runtime library. However, to aid development, it can be useful
for users to produce one.

This sample illustrates both modes of use. It may eventually be included in
the IREE repository proper as a standalone project.

## Options

* `-DIREE_ROOT_DIR=<path>` : Override the path to the main IREE repo. Defaults
  to assuming that `iree` is checked out adjacent to `iree-samples`.
* `-DBUILD_SHARED_LIBS=ON` : Builds a libireert.so (or corresponding DLL/dylib)
  for development use. Note that the low-level IREE runtime API is fine grained
  and usage in a shared library will pessimize optimizations. Therefore, this
  is only recommended for development or getting started.
* `-DIREERT_ENABLE_LTO=ON` : Enables LTO if the toolchain supports it. This is
  supported for both shared and static library builds, but for shared libraries,
  the optimizations stop at the exported symbol boundary. As of early 2023,
  this has the side effect of reducing the binary size by ~16%.

## Using in external builds

When built, `lib/` and `include/` directories will be populated. It should
be possible to use these with no further manipulation. While using the main
`iree` project as a sub-project is recommended for those who can, using
standalone headers and libraries is expected to benefit language bindings and
other indirect uses of the API, since such tooling often interops at the
C/library level, not the build system level.

## Typical use

```
cmake -GNinja -Bbuild .
cmake --build build
./build/bin/ireert_test
```

If we run the above commands on macOS, we will get
`build/lib/ireert.framework`, which includes the static library and
header files.

## Build for iOS

Download Xcode from the App Store.  Stick with the Python that comes with Xcode.

Don't try to be clever by downloading a specific version of Xcode from https://developer.apple.com/download or installing Python from python.org. Homebrew may download Xcode Command Line Tools, however this is inadequate to develop IREE for iOS, tvOS, and watchOS.

Install [Homebrew](https://brew.sh/) after installing Xcode.

Install build tools.

```bash
brew install cmake ninja tree
```

Git-clone IREE to a directory side-by-side with this repository.
```bash
cd ~/work # Where we have ~/work/iree-samples
git clone --recursive https://github.com/openxla/iree
cd iree
git submodule update --init
```

Run the bash script `create_xcframework.sh` to build
1. the IREE compiler, runtime, and Python binding for macOS, and
1. the IREE runtime for macOS and iOS.


If you want Metal GPU, please add the  `-m` option.
```bash
./create_xcframework.sh -m
```

Currently, IREE works with macOS and iOS.  We are working on enabling IREE for tvOS and watchOS.
