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

Follow [this step](https://iree-org.github.io/iree/building-from-source/ios/#build-the-iree-compiler-for-the-host) to build the IREE compiler for macOS.
Let's say the CMake build directory is `$HOME/w/iree/build`. To configure and build this project for the iOS Simulator, we could use the following commands.

```
cmake -S . -B build-ios-sim -GNinja \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=$(xcodebuild -version -sdk iphonesimulator Path) \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_SYSTEM_PROCESSOR=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DCMAKE_IOS_INSTALL_COMBINED=YES \
  -DIREE_HOST_BIN_DIR="$HOME/w/iree-build/install/bin" \
  -DCMAKE_INSTALL_PREFIX=../build-ios-sim/install \
  -DIREE_BUILD_COMPILER=OFF
  
cmake --build build-ios-sim
```

This will give us the app bundle `build-ios-sim/bin/ireert_test.app` and the IREE runtime framework `build-ios-sim/lib/ireert.framework`.

To configure and build for iOS devices, we can change `-DCMAKE_OSX_SYSROOT=$(xcodebuild -version -sdk iphonesimulator Path)` into `-DCMAKE_OSX_SYSROOT=$(xcodebuild -version -sdk iphoneos Path)`.
