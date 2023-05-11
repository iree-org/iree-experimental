#!/bin/bash

# exit when any command fails
set -e

# Force build to use this version of the command line tools
# Overrides what `xcode-select -p` currently points to
export DEVELOPER_DIR=/Applications/Xcode.app

dbg="-DCMAKE_BUILD_TYPE=Release"
metal=""

# The -f option forces the rebuild of IREE runtime frameworks.  The -r
# option forces the rebuild of both the compiler and the runtime.
while getopts dDhfrsm flag; do
    case "${flag}" in
	d) dbg="-DCMAKE_BUILD_TYPE=RelWithDebInfo";;
	D) dbg="-DCMAKE_BUILD_TYPE=Debug";;
	s) sans="-DIREE_ENABLE_TSAN=ON -DIREE_BYTECODE_MODULE_ENABLE_TSAN=ON -DIREE_BYTECODE_MODULE_FORCE_LLVM_SYSTEM_LINKER=ON";;
	m) metal="-DIREE_EXTERNAL_HAL_DRIVERS=metal";;
	r) FORCE_REBUILD_COMPILER="YES"
	    FORCE_REBUILD_RUNTIME="YES";;
	f) FORCE_REBUILD_COMPILER="NO";
	   FORCE_REBUILD_RUNTIME="YES" ;;
	*)
            echo "$0 -h|-f"
            echo "  -h : display this message"
            echo "  -f : force rebuild runtime frameworks"
            exit 0
            ;;
    esac
done

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
IREE_SRC_DIR=$(realpath "$SCRIPT_DIR"/../../iree) # The IREE project must be side-by-side with this project.
IREE_BUILD_DIR=$SCRIPT_DIR/build
IREE_BUILD_COMPILER_DIR=$IREE_BUILD_DIR/compiler
IREE_BUILD_COMPILER_INSTALL_DIR=$IREE_BUILD_COMPILER_DIR/install
IREE_BUILD_RUNTIME_DIR=$IREE_BUILD_DIR/runtime
IREE_BUILD_RUNTIME_XCFRAMEWORK="$IREE_BUILD_RUNTIME_DIR"/iree.xcframework

function build_iree_for_host() {
    # Build IREE compiler, runtime, and Python bindings for the host, if not yet.
    if [[ ! -x "$IREE_BUILD_COMPILER_INSTALL_DIR"/bin/iree-compile ]]; then
        echo "┌------------------------------------------------------------------------------┐"
        echo "  Building IREE for the host ... "
        echo "   src: $IREE_SRC_DIR "
        echo "   build: $IREE_BUILD_COMPILER_DIR "
        echo "   log: $IREE_BUILD_COMPILER_DIR/build.log"
        echo "└------------------------------------------------------------------------------┘"

	mkdir -p "$IREE_BUILD_COMPILER_DIR" # So to create the build.log file.
	
        # Install deps required to build the Python binding.
        python3 -m pip install \
	       -r "$IREE_SRC_DIR"/runtime/bindings/python/iree/runtime/build_requirements.txt \
	       >"$IREE_BUILD_COMPILER_DIR"/build.log 2>&1

        cmake -S "$IREE_SRC_DIR" \
              -B "$IREE_BUILD_COMPILER_DIR" \
	      -DCMAKE_C_COMPILER=/usr/bin/clang \
	      -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
	      -DCMAKE_RANLIB=/usr/bin/ranlib \
	      -DCMAKE_C_COMPILER_RANLIB=/usr/bin/ranlib \
	      -DCMAKE_CXX_COMPILER_RANLIB=/usr/bin/ranlib \
	      -DCMAKE_AR=/usr/bin/ar \
	      "$dbg" \
	      "$sans" \
	      "$metal" \
              -GNinja \
              -DIREE_BUILD_TESTS=OFF \
              -DIREE_BUILD_SAMPLES=OFF \
              -DIREE_BUILD_PYTHON_BINDINGS=ON \
              -DPython3_EXECUTABLE="$(which python3)" \
              -DIREE_BUILD_TRACY=OFF \
              -DCMAKE_INSTALL_PREFIX="$IREE_BUILD_COMPILER_INSTALL_DIR" \
	      > "$IREE_BUILD_COMPILER_DIR"/build.log 2>&1
        cmake --build "$IREE_BUILD_COMPILER_DIR" --target install \
              >> "$IREE_BUILD_COMPILER_DIR"/build.log 2>&1
    fi
}

# Re-build the IREE runtime into a framework for each target.
if [[ $FORCE_REBUILD_COMPILER == "YES" ]]; then
    echo "┌------------------------------------------------------------------------------┐"
    echo "  Deleting existig IREE compiler ..."
    echo "└------------------------------------------------------------------------------┘"
    rm -rf "$IREE_BUILD_COMPILER_DIR"
fi

if [[ $FORCE_REBUILD_RUNTIME == "YES" ]]; then
    echo "┌------------------------------------------------------------------------------┐"
    echo "  Deleting existig IREE runtime frameworks ..."
    echo "└------------------------------------------------------------------------------┘"
    rm -rf "$IREE_BUILD_RUNTIME_DIR"
fi

function readlink_if_symbolic() {
    # NOTE: In macOS framework bundles, the library is a symbolic
    # link. Whereas in iOS framework bundles, libraries are libraries.
    file=$1
    if [[ -L $file ]]; then
        echo "$(dirname "$file")/$(readlink "$file")"
    else
        echo "$file"
    fi
}

# TODO(wangkuiyi) The IREE runtime depends on third-party libraries
# including flatcc and cpuinfo.  CMake (and most build tools) do not
# merge all these static libraries into one before making frameworks.
# Instead, when bulding applications, CMake will link all these static
# libraries.  However, iOS apps are not usually built with CMake.
# Instead, developers prefer to use Xcode.  This following steps merge
# all static libraries in a very hacky way.  We will need a cleaner
# solution.
#
# Copy the following static libraries from the CMake build directory
# into the corresponding target framework in the XCFramework.
#
#  $CMAKE_BUILD_DIR/iree_core/third_party/cpuinfo/libcpuinfo.a
#  $CMAKE_BUILD_DIR/iree_core/third_party/cpuinfo/deps/clog/libclog.a
#  $CMAKE_BUILD_DIR/iree_core/build_tools/third_party/flatcc/libflatcc_parsing.a
#
function merge_static_libraries() {
    label=$1
    build_dir="$IREE_BUILD_RUNTIME_DIR"/"$label"

    echo "┌------------------------------------------------------------------------------┐"
    echo "  Build all-in-one static library for $label ..."
    echo "   log: $build_dir/merge_static_libraries.log"
    echo "└------------------------------------------------------------------------------┘"

    static_lib=$(readlink_if_symbolic "$build_dir"/lib/iree.framework/iree)
    static_lib_dir=$(dirname "$static_lib")

    for sl in $(du -a "$build_dir"/iree_core | grep '\.a$' | cut -f 2); do
        cp "$sl" "$static_lib_dir"
    done

    (
        cd "$static_lib_dir"
        mv iree libiree.a
        if libtool -static -o iree libiree.a libflatcc_parsing.a libclog.a libcpuinfo.a \
            >"$build_dir"/merge_static_libraries.log 2>&1; then
            echo "Merged for $label successfully"
        fi
    )
}

function build_iree_runtime_for_device() {
    case $1 in
    ios-sim) sysname=iOS; sdk=iphonesimulator ;;
    ios-dev) sysname=iOS; sdk=iphoneos ;;
    tv-sim) sysname=tvOS; sdk=appletvsimulator ;;
    tv-dev) sysname=tvOS; sdk=appletvos ;;
    watch-sim) sysname=watchOS; sdk=watchsimulator ;;
    watch-dev) sysname=watchOS; sdk=watchos ;;
    *)
        echo "Error: Unknown target $1"
        exit 5
        ;;
    esac

    if [[ "$sans" != "" ]]; then	# Users chose to enable TSAN
	if [[ "$sdk" == "iphonesimulator" ]]; then
	    sans="$sans -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread"
	else
	    sans="" 		# devices do not support TSAN
	fi
    fi

    arch=$2
    sysarch=$arch
    if [[ "$sysarch" == "arm64e" ]]; then
	sysarch=arm64 # pytorch/cpuinfo does not recognize CMAKE_SYSTEM_PROCESSOR=arm64e
    fi

    label=$1-"$arch"
    build_dir="$IREE_BUILD_RUNTIME_DIR"/"$label"

    test_file="$build_dir"/lib/iree.framework/iree
    if test -f "$test_file" && lipo -info "$test_file"; then
        echo "Skip building iree.framework for $label."
    else
        echo "┌------------------------------------------------------------------------------┐"
        echo "  Building IREE runtime for $label ..."
        echo "   src: $IREE_SRC_DIR "
        echo "   build: $build_dir "
        echo "   build log: $build_dir/build.log"
        echo "└------------------------------------------------------------------------------┘"
        mkdir -p "$build_dir" # So to create the build.log file.
        cmake -S . \
              -B "$build_dir" \
              -DIREE_ROOT_DIR="$IREE_SRC_DIR" \
	      -DCMAKE_C_COMPILER=/usr/bin/clang \
	      -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
	      -DCMAKE_RANLIB=/usr/bin/ranlib \
	      -DCMAKE_C_COMPILER_RANLIB=/usr/bin/ranlib \
	      -DCMAKE_CXX_COMPILER_RANLIB=/usr/bin/ranlib \
	      -DCMAKE_AR=/usr/bin/ar \
	      "$dbg" \
	      "$sans" \
	      "$metal" \
	      -GNinja \
              -DCMAKE_SYSTEM_NAME=$sysname \
              -DCMAKE_OSX_SYSROOT="$(xcodebuild -version -sdk $sdk Path)" \
              -DCMAKE_OSX_ARCHITECTURES="$arch" \
              -DCMAKE_SYSTEM_PROCESSOR="$sysarch" \
              -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
              -DCMAKE_IOS_INSTALL_COMBINED=YES \
              -DIREE_HOST_BIN_DIR="$IREE_BUILD_COMPILER_INSTALL_DIR"/bin \
              -DCMAKE_INSTALL_PREFIX="$build_dir"/install \
              -DIREE_BUILD_COMPILER=OFF >"$build_dir"/build.log 2>&1
        cmake --build "$build_dir" >>"$build_dir"/build.log 2>&1

        merge_static_libraries "$label"
    fi
}

function build_iree_runtime_for_macos() {
    arch=$1
    sysarch=$arch
    if [[ "$sysarch" == "arm64e" ]]; then
	sysarch=arm64 # pytorch/cpuinfo does not recognize CMAKE_SYSTEM_PROCESSOR=arm64e
    fi
    label=macos-"$arch"
    build_dir="$IREE_BUILD_RUNTIME_DIR"/"$label"

    if [[ "$sans" != "" ]]; then	# Users chose to enable TSAN
	sans="$sans -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread"
    fi

    test_file="$build_dir"/lib/iree.framework/iree
    if test -f "$test_file" && lipo -info "$test_file"; then
        echo "Skip building iree.framework for $label."
    else
        echo "┌------------------------------------------------------------------------------┐"
        echo "  Building for $label ..."
        echo "   src: $IREE_SRC_DIR "
        echo "   build: $build_dir "
        echo "   build log: $build_dir/build.log"
        echo "└------------------------------------------------------------------------------┘"
        mkdir -p "$build_dir" # So to create the build.log file.
        cmake -S . \
            -B "$build_dir" \
            -DIREE_ROOT_DIR="$IREE_SRC_DIR" \
	    -DCMAKE_C_COMPILER=/usr/bin/clang \
	    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
	    -DCMAKE_RANLIB=/usr/bin/ranlib \
	    -DCMAKE_C_COMPILER_RANLIB=/usr/bin/ranlib \
	    -DCMAKE_CXX_COMPILER_RANLIB=/usr/bin/ranlib \
	    -DCMAKE_AR=/usr/bin/ar \
	    "$dbg" \
	    "$sans" \
	    "$metal" \
            -GNinja \
            -DCMAKE_OSX_ARCHITECTURES="$arch" \
	    -DCMAKE_SYSTEM_PROCESSOR="$sysarch" \
	    > "$build_dir"/build.log 2>&1
        cmake --build "$build_dir" >> "$build_dir"/build.log 2>&1

        merge_static_libraries "$label"
    fi
}

function merge_fat_static_library() {
    src_label=$2
    dst_label=$1

    src="$IREE_BUILD_RUNTIME_DIR"/$src_label/lib/iree.framework/iree
    dst="$IREE_BUILD_RUNTIME_DIR"/$dst_label/lib/iree.framework/iree

    if lipo -info "$dst" | grep 'Non-fat' >/dev/null; then
        echo "┌------------------------------------------------------------------------------┐"
        echo "  Building FAT static library ..."
        echo "   src: $src"
        echo "   dst: $dst"
        echo "└------------------------------------------------------------------------------┘"
        merged=/tmp/libmerged-"$src_label"-"$dst_label".a
        lipo "$src" "$dst" -create -output "$merged"
        mv "$merged" "$dst"
    fi
}



# Step 0. Build IREE compiler, runtime, and Python bindings for the host (macOS).
build_iree_for_host

# Step 1. Build the IREE runtime into the following frameworks
#
# Note: We cannot build for dev-x86_64 because Apple does not offer
# SDK for it. If we do so, CMake will prompt us about missing required
# architecture x86_64 in file
# /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS16.2.sdk/usr/lib/libc++.tbd
# build_iree_runtime_for_ios dev x86_64
#
# This step also merge dependent static libraries into the target library.
build_iree_runtime_for_device ios-sim arm64
build_iree_runtime_for_device ios-sim x86_64
build_iree_runtime_for_device ios-dev arm64
build_iree_runtime_for_device ios-dev arm64e
build_iree_runtime_for_macos x86_64
build_iree_runtime_for_macos arm64

# Step 2. Merge the frameworks of the same OS platform
merge_fat_static_library ios-sim-arm64 ios-sim-x86_64
merge_fat_static_library ios-dev-arm64 ios-dev-arm64e
merge_fat_static_library macos-arm64 macos-x86_64

# Step 3. Merge the above frameworks into an XCFramework
echo "┌------------------------------------------------------------------------------┐"
echo "  Aggregating frameworks into an xcframework ..."
echo "└------------------------------------------------------------------------------┘"
rm -rf "$IREE_BUILD_RUNTIME_XCFRAMEWORK"
# TODO(wangkuiyi) xcodebuild cannot accept input frameworks built with
# LTO enabled.  It will complain unknown architecture, even if lipo
# -info could identify the architecture.  Currently, we disable LTO
# for macOS/iOS.  But LTO should be enabled for smaller binary size
# and better runtime performance.
xcodebuild -create-xcframework \
    -framework "$IREE_BUILD_RUNTIME_DIR"/macos-arm64/lib/iree.framework \
    -framework "$IREE_BUILD_RUNTIME_DIR"/ios-sim-arm64/lib/iree.framework \
    -framework "$IREE_BUILD_RUNTIME_DIR"/ios-dev-arm64/lib/iree.framework \
    -output "$IREE_BUILD_RUNTIME_XCFRAMEWORK"
tree -L 1 -d "$IREE_BUILD_RUNTIME_XCFRAMEWORK"
