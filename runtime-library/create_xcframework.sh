#!/bin/bash

# exit when any command fails
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
# TODO(wangkuiyi): Here we assume that we git cloned iree-org/iree and
# iree-org/iree-samples to the same directory.  We might want to make
# iree-org/iree a git submodule of the repo that builds the IREE
# runtime.
IREE_HOST_BUILD_DIR=$SCRIPT_DIR/../../iree-build/install/bin

if [[ ! -x $IREE_HOST_BUILD_DIR/iree-compile ]]; then
    echo "Please build IREE for the host before running this script."
    echo "https://iree-org.github.io/iree/building-from-source/ios/#configure-and-build"
    exit 1
fi

# Re-build the IREE runtime into a framework for each target.
rm -rf build*

# TODO(wangkuiyi): The following CMake configuration and building
# steps assume that we run this script on macOS with Xcode and iOS
# SDK.  We should add checks for these prerequisites.
#
echo "===================================================================="
echo "Building for the host ..."
echo "===================================================================="
cmake -GNinja -Bbuild . &&
cmake --build build

# TODO(wangkuiyi): The following CMake configuration specify build for
# iOS Sumulator running on ARM64.  We might need to add more
# architectures for users using x86_64 Macs, like Mac Pro 2019.
echo "===================================================================="
echo "Building for the iOS Simulator ..."
echo "===================================================================="
cmake -S . -B build-ios-sim -GNinja \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=$(xcodebuild -version -sdk iphonesimulator Path) \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_SYSTEM_PROCESSOR=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DCMAKE_IOS_INSTALL_COMBINED=YES \
  -DIREE_HOST_BIN_DIR="$HOME/w/iree-build/install/bin" \
  -DCMAKE_INSTALL_PREFIX=../build-ios-sim/install \
  -DIREE_BUILD_COMPILER=OFF &&  
cmake --build build-ios-sim

echo "===================================================================="
echo "Building for iOS devices ..."
echo "===================================================================="
cmake -S . -B build-ios-dev -GNinja \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=$(xcodebuild -version -sdk iphoneos Path) \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_SYSTEM_PROCESSOR=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DCMAKE_IOS_INSTALL_COMBINED=YES \
  -DIREE_HOST_BIN_DIR="$HOME/w/iree-build/install/bin" \
  -DCMAKE_INSTALL_PREFIX=../build-ios-dev/install \
  -DIREE_BUILD_COMPILER=OFF &&  
cmake --build build-ios-dev

echo "===================================================================="
echo "Aggregating frameworks into an xcframework ..."
echo "===================================================================="
rm -rf iree.xcframework
# TODO(wangkuiyi) xcodebuild cannot accept input frameworks built with
# LTO enabled.  It will complain unknown architecture, even if lipo
# -info could identify the architecture.  Currently, we disable LTO
# for macOS/iOS.  But LTO should be enabled for smaller binary size
# and better runtime performance.
xcodebuild -create-xcframework \
 	   -framework build/lib/iree.framework \
 	   -framework build-ios-sim/lib/iree.framework \
 	   -framework build-ios-dev/lib/iree.framework \
	   -output iree.xcframework

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
echo "===================================================================="
echo "Build all-in-one static library for the iOS Simulator ..."
echo "===================================================================="
for sl in $(du -a build-ios-sim/iree_core | grep '\.a$' | cut -f 2); do
    cp $sl iree.xcframework/ios-arm64-simulator/iree.framework/
done
(
    cd iree.xcframework/ios-arm64-simulator/iree.framework/
    # Merge the original iree static library with its dependencies
    # into the new iree static library.
    mv iree libiree.a
    libtool -static -o iree libiree.a libflatcc_parsing.a libclog.a libcpuinfo.a
)

echo "===================================================================="
echo "Build all-in-one static library for iOS devices ..."
echo "===================================================================="
for sl in $(du -a build-ios-dev/iree_core | grep '\.a$' | cut -f 2); do
    cp $sl iree.xcframework/ios-arm64/iree.framework/
done
(
    cd iree.xcframework/ios-arm64-simulator/iree.framework/
    # Merge the original iree static library with its dependencies
    # into the new iree static library.
    mv iree libiree.a
    libtool -static -o iree libiree.a libflatcc_parsing.a libclog.a libcpuinfo.a
)

echo "===================================================================="
echo "Build all-in-one static library for macOS ..."
echo "===================================================================="
for sl in $(du -a build/iree_core | grep '\.a$' | cut -f 2); do
    cp $sl iree.xcframework/macos-arm64/iree.framework/Versions/Current/
done
(
    cd iree.xcframework/macos-arm64/iree.framework/Versions/Current/
    # Merge the original iree static library with its dependencies
    # into the new iree static library.
    mv iree libiree.a
    libtool -static -o iree libiree.a libflatcc_parsing.a libclog.a libcpuinfo.a
)
