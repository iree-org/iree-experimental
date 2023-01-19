#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
IREE_HOST_BUILD_DIR=$SCRIPT_DIR/../../iree-build/install/bin

if [[ ! -x $IREE_HOST_BUILD_DIR/iree-compile ]]; then
    echo "Please build IREE for the host https://iree-org.github.io/iree/building-from-source/ios/#configure-and-build"
    exit 1
fi

echo "Building for the host ..."
cmake -GNinja -Bbuild . &&
cmake --build build

echo "Building for the iOS Simulator ..."
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

echo "Building for iOS devices ..."
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

echo "Aggregating frameworks into an xcframework ..."
rm -rf iree.xcframework
xcodebuild -create-xcframework \
 	   -framework build/lib/iree.framework \
 	   -framework build-ios-sim/lib/iree.framework \
 	   -framework build-ios-dev/lib/iree.framework \
	   -output iree.xcframework

