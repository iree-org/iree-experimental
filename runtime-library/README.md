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

## Options:

* `-DBUILD_SHARED_LIBS=ON` : Builds a libireert.so (or corresponding DLL/dylib)
  for development use. Note that the low-level IREE runtime API is fine grained
  and usage in a shared library will pessimize optimizations. Therefore, this
  is only recommended for development or getting started.
* `-DIREERT_ENABLE_LTO=ON` : Enables LTO if the toolchain supports it. This is
  supported for both shared and static library builds, but for shared libraries,
  the optimizations stop at the exported symbol boundary. As of early 2023,
  this has the side effect of reducing the binary size by ~16%.

## Using in external builds.

When built, `lib/` and `include/` directories will be populated. It should
be possible to use these with no further manipulation. While using the main
`iree` project as a sub-project is recommended for those who can, using
standalone headers and libraries is expected to benefit language bindings and
other indirect uses of the API, since such tooling often interops at the
C/library level, not the build system level.

## Typical use

```
cmake -GNinja -Bbuild .
cd build
ninja
./bin/ireert_test
```
