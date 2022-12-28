#!/bin/bash

################################################################################
# IREE + transform dialect helpers.
################################################################################

function iree-transform-get-args() {
  usage() { 
    echo 1>&2 'Usage: iree-transform-xxx <mlir-input-file> -b <backend> [-c <codegen-spec-file>] [-d <dispatch-spec-file>] [-- extra arguments]'
  }
  
  MLIR_FILE=$1
  local OPTIND o BACKEND CODEGEN_SPEC_FILE DISPATCH_SPEC_FILE
  OPTIND=2
  while getopts ":c:d:b:" o; do
    case "${o}" in
      b)
        BACKEND=${OPTARG}
        ;;
      c)
        CODEGEN_SPEC_FILE=${OPTARG}
        ;;
      d)
        DISPATCH_SPEC_FILE=${OPTARG}
        ;;
      *)
        opts+=("-${opt}"); [[ -n "$OPTARG" ]] && opts+=("$OPTARG")
        ;;
    esac
  done
  shift $(expr $OPTIND - 1 )

  if [ -z "${BACKEND+x}" ] || [ -z "${MLIR_FILE+x}" ] ; then
    usage
    return 1
  fi
 
  MAYBE_EMPTY_CODEGEN_SPEC_FILE=${CODEGEN_SPEC_FILE:=/dev/null}
  MAYBE_EMPTY_DISPATCH_SPEC_FILE=${DISPATCH_SPEC_FILE:=/dev/null}
  # For debugging purposes
  # echo BACKEND=~~~${BACKEND}~~~MLIR_FILE=~~~${MLIR_FILE}~~~\
  # CODEGEN_SPEC_FILE=~~~${CODEGEN_SPEC_FILE}~~~\
  # MAYBE_EMPTY_DISPATCH_SPEC_FILE=~~~${MAYBE_EMPTY_DISPATCH_SPEC_FILE}~~~\
  # REST=~~~${@}~~~
  echo ${BACKEND} ${MLIR_FILE} ${MAYBE_EMPTY_CODEGEN_SPEC_FILE} ${MAYBE_EMPTY_DISPATCH_SPEC_FILE} ${@}
}

# Example usage:
# iree-transform-opt-dispatch-only tests/transform_dialect/cpu/matmul.mlir \
#   -b llvm-cpu \
#   [ -c ./tests/transform_dialect/cpu/matmul_codegen_spec.mlir ] \
#   [ -d ./tests/transform_dialect/cpu/matmul_dispatch_spec.mlir ] \
#   [ -- extra_stuff ]
function iree-transform-opt-dispatch-only() {
  ARGS=$(iree-transform-get-args $@)
  if [ $? -ne 0 ]; then
    return 1
  fi
  read -r BACKEND MLIR_FILE CODEGEN_SPEC_FILE DISPATCH_SPEC_FILE EXTRA_ARGS <<<$(echo ${ARGS})
  
  if test ${DISPATCH_SPEC_FILE} == /dev/null; then
    DISPATCH_FLAG="--iree-flow-enable-aggressive-fusion"
  else
    DISPATCH_FLAG="--iree-flow-dispatch-use-transform-dialect=${DISPATCH_SPEC_FILE}"
  fi

  iree-opt ${MLIR_FILE} \
    --iree-abi-transformation-pipeline \
    --iree-flow-transformation-pipeline \
    ${DISPATCH_FLAG} \
    ${EXTRA_ARGS}
}

# Example usage:
# iree-transform-opt tests/transform_dialect/cpu/matmul.mlir \
#   -b llvm-cpu \
#   [ -c ./tests/transform_dialect/cpu/matmul_codegen_spec.mlir ] \
#   [ -d ./tests/transform_dialect/cpu/matmul_dispatch_spec.mlir ] \
#   [ -s cpu-matmul-strategy ]
#   [ extra_stuff ]
function iree-transform-opt() {
  ARGS=$(iree-transform-get-args $@)
  if [ $? -ne 0 ]; then
    return 1
  fi
  read -r BACKEND MLIR_FILE CODEGEN_SPEC_FILE DISPATCH_SPEC_FILE EXTRA_ARGS <<<$(echo ${ARGS})
  
  if [ -z "${DUMP_TRANSFORM_REPRO+x}" ]
  then
    TRANSFORM_REPRO_FLAG=""
  else
    TRANSFORM_REPRO_FLAG="--debug-only=iree-transform-dialect-save-repro"
  fi

  if test ${BACKEND} == cuda; then
    CODEGEN_FLAG="--pass-pipeline=builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))"
    if test ${CODEGEN_SPEC_FILE} == /dev/null; then
      CODEGEN_FLAG=${CODEGEN_FLAG}" --iree-codegen-llvmgpu-enable-transform-dialect-jit"
    else
      CODEGEN_FLAG=${CODEGEN_FLAG}" --iree-codegen-llvmgpu-use-transform-dialect=${CODEGEN_SPEC_FILE}"
    fi
    CODEGEN_FLAG="${CODEGEN_FLAG} ${TRANSFORM_REPRO_FLAG}"
  elif test ${BACKEND} == llvm-cpu; then
    CODEGEN_FLAG="--pass-pipeline=builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))"
    if test ${CODEGEN_SPEC_FILE} == /dev/null; then
      CODEGEN_FLAG=${CODEGEN_FLAG}" --iree-codegen-llvmcpu-enable-transform-dialect-jit"
    else
      CODEGEN_FLAG=${CODEGEN_FLAG}" --iree-codegen-llvmcpu-use-transform-dialect=${CODEGEN_SPEC_FILE}"
    fi
    CODEGEN_FLAG="${CODEGEN_FLAG} ${TRANSFORM_REPRO_FLAG}"
  else
    echo "Unknown IREE backend: " ${BACKEND}
    return 1
  fi

  iree-transform-opt-dispatch-only ${MLIR_FILE} -b ${BACKEND} -c ${CODEGEN_SPEC_FILE} -d ${DISPATCH_SPEC_FILE} | \
  iree-opt --iree-hal-target-backends=${BACKEND} --iree-stream-transformation-pipeline --iree-hal-configuration-pipeline | \
  iree-opt ${CODEGEN_FLAG} ${EXTRA_ARGS}
}

# Example usage:
# Pipe this through iree-run-module, e.g.:
#
#   iree-transform-compile -b llvm-cpu -i i.mlir -c c.mlir -d d.mlir -- \
#       --iree-llvm-target-triple=x86_64-pc-linux-gnu --iree-llvm-target-cpu-features=host | \
#     iree-run-module --entry_function=max_sub_exp --device=local-task \
#       --function_input="32x1024xf32=1" --function_input="32xf32=-1066"
#
#   iree-transform-compile -b llvm-cpu -i i.mlir -c c.mlir -d d.mlir -- \
#       --iree-llvm-target-triple=x86_64-pc-linux-gnu --iree-llvm-target-cpu-features=host --iree-hal-benchmark-dispatch-repeat-count=10 | \
#     iree-benchmark-module --device=local-task --task_topology_group_count=0 \
#       --batch_size=10 --entry_function=reduce \
#       --function_input="32x1024xf32=1" --function_input="32xf32=-1066"
#
function iree-transform-compile() {
  ARGS=$(iree-transform-get-args $@)
  if [ $? -ne 0 ]; then
    return 1
  fi
  read -r BACKEND MLIR_FILE CODEGEN_SPEC_FILE DISPATCH_SPEC_FILE EXTRA_ARGS <<<$(echo ${ARGS})

  if [ -z "${DUMP_TRANSFORM_REPRO+x}" ]
  then
    TRANSFORM_REPRO_FLAG=""
  else
    TRANSFORM_REPRO_FLAG="--debug-only=iree-transform-dialect-save-repro"
  fi

  if test ${BACKEND} == "cuda"; then
    if test ${CODEGEN_SPEC_FILE} == /dev/null; then
      CODEGEN_FLAG="--iree-codegen-llvmgpu-enable-transform-dialect-jit"
    else
      CODEGEN_FLAG="--iree-codegen-llvmgpu-use-transform-dialect=${CODEGEN_SPEC_FILE}"
    fi
    CODEGEN_FLAG="${CODEGEN_FLAG} ${TRANSFORM_REPRO_FLAG}"
  elif test ${BACKEND} == "llvm-cpu"; then
    if test ${CODEGEN_SPEC_FILE} == /dev/null; then
      CODEGEN_FLAG="--iree-codegen-llvmcpu-enable-transform-dialect-jit"
    else
      CODEGEN_FLAG="--iree-codegen-llvmcpu-use-transform-dialect=${CODEGEN_SPEC_FILE}"
    fi
    CODEGEN_FLAG="${CODEGEN_FLAG} ${TRANSFORM_REPRO_FLAG}"
  else
    echo "Unknown IREE backend: " ${BACKEND}
    return 1
  fi
  if test ${DISPATCH_SPEC_FILE} == /dev/null; then
    DISPATCH_FLAG="--iree-flow-enable-aggressive-fusion"
  else
    DISPATCH_FLAG="--iree-flow-dispatch-use-transform-dialect=${DISPATCH_SPEC_FILE}"
  fi

  iree-compile ${MLIR_FILE} --iree-hal-target-backends=${BACKEND} \
    ${DISPATCH_FLAG} \
    ${CODEGEN_FLAG} \
    ${EXTRA_ARGS}
}

################################################################################
# Basic benchmarking helpers.
################################################################################
function get-p50-from-nvprof() {
  if (($# < 3)); then
    echo "Usage: get-p50-from-nvprof nvprof_trace_string function_name num_iterations"
    return 1
  fi

  NVPROF_TRACE=$1
  FUNCTION_NAME=$2
  NUM_ITERATIONS=$3


  P50_TIME=$(echo "${NVPROF_TRACE}" | \
    grep ${FUNCTION_NAME} | \
    awk {'print $2'} | \
    tail -n ${NUM_ITERATIONS} | \
    sed "s/us/*1000/g" | \
    sed "s/ms/*1000000/g" | \
    sed "s/s/*1000000000/g" | \
    bc | \
    sort --version-sort | \
    head -n $(echo ${NUM_ITERATIONS}/2 | bc) | \
    tail -n 1)

  echo ${P50_TIME}
}

################################################################################
# Functions below instantiate IR from a stub by plugging desired sizes and 
# provide reproducers for the interpreted mode.
################################################################################
function benchmark-transform-create() {
  cmake --build ./build --target iree-opt iree-compile iree-run-module

  if [[ $# < 4 || $# > 5 ]]; then
    echo "Usage: benchmark-transform-run-nvprof stub_file_name function_name SZ1 SZ2 [optional run keyword]"
    return 1
  fi
  TRANSFORM_DIALECT_BENCHMARK_STUB_FILE=$1; shift
  FUNCTION_NAME=$1; shift
  SZ1=$1; shift
  SZ2=$1; shift
  if [[ $# == 1 ]]; then
    RUN=$1
  fi

  TRANSFORM_DIALECT_SOURCE_FILE=/tmp/${FUNCTION_NAME}_${SZ1}x${SZ2}.mlir
  # Extract exactly the func we care about and let `mlir-opt -symbol-dce ` clean
  # up the rest of the IR.
  # This lets us use files with multiple funcs
  cat ${TRANSFORM_DIALECT_BENCHMARK_STUB_FILE} | \
    sed "s/private @${FUNCTION_NAME}(/@${FUNCTION_NAME}(/g" | \
    sed "s/\${SZ1}/${SZ1}/g" | \
    sed "s/\${SZ2}/${SZ2}/g" | \
    mlir-opt -symbol-dce > ${TRANSFORM_DIALECT_SOURCE_FILE}

  echo iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda  -- --mlir-disable-threading 2>&1 > /dev/null || exit 1
  iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda  -- --mlir-disable-threading 2>&1 > /dev/null || exit 1
  TRANSFORM_DIALECT_TRANSFORM_FILE=$(DUMP_TRANSFORM_REPRO=1 iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda  -- --mlir-disable-threading 2>&1 > /dev/null | grep iree-opt | awk '{print $4}')
  
  if [[ ${TRANSFORM_DIALECT_TRANSFORM_FILE} =~ "tmp" ]] ; then
    : # noop
  else
    echo "Could not create TRANSFORM_DIALECT_TRANSFORM_FILE (match failure or compile failure)?"
    echo Try running:    iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda  -- --mlir-disable-threading
    return 1
  fi

  echo ==========================================================
  echo Problem created successufully, reproduction instructions:
  echo ==========================================================
  echo Transform dialect source file is: ${TRANSFORM_DIALECT_SOURCE_FILE}
  echo Transform dialect transform file is: ${TRANSFORM_DIALECT_TRANSFORM_FILE}
  echo Dump transformed IR with: benchmark-transform-run-iree-opt ${TRANSFORM_DIALECT_SOURCE_FILE} ${TRANSFORM_DIALECT_TRANSFORM_FILE}
  echo Dump transformed PTX with: benchmark-transform-run-iree-compile ${TRANSFORM_DIALECT_SOURCE_FILE} ${TRANSFORM_DIALECT_TRANSFORM_FILE}
  echo Run nvprof with e.g.: benchmark-transform-run-nvprof ${TRANSFORM_DIALECT_SOURCE_FILE} ${TRANSFORM_DIALECT_TRANSFORM_FILE} ${FUNCTION_NAME} ${SZ1} ${SZ2}
  echo ==========================================================

  if [[ ${RUN} == "run" ]]; then
    benchmark-transform-run-nvprof ${TRANSFORM_DIALECT_SOURCE_FILE} ${TRANSFORM_DIALECT_TRANSFORM_FILE} ${FUNCTION_NAME} ${SZ1} ${SZ2}
  fi
}

function benchmark-transform-run-iree-opt() {
  cmake --build ./build --target iree-opt

  if (($# != 2)); then
    echo "Usage: benchmark-transform-run-iree-opt source-file transform-file [optional extra args]"
    return 1
  fi

  TRANSFORM_DIALECT_SOURCE_FILE=$1; shift
  TRANSFORM_DIALECT_TRANSFORM_FILE=$1; shift
  echo iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} -- $@
  iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} -- $@
}

function benchmark-transform-run-iree-compile() {
  cmake --build ./build --target iree-compile

  if (($# != 2)); then
    echo "Usage: benchmark-transform-run-iree-compile source-file transform-file [optional extra args]"
    return 1
  fi

  TRANSFORM_DIALECT_SOURCE_FILE=$1; shift
  TRANSFORM_DIALECT_TRANSFORM_FILE=$1; shift
  echo iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda -c ${TRANSFORM_DIALECT_TRANSFORM_FILE}
  iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} -- $@
}

function benchmark-transform-run-nvprof() {
  cmake --build ./build --target iree-compile

  if (($# < 5)); then
    echo "Usage: benchmark-transform-run-nvprof source-file transform-file function_name SZ1 SZ2"
    return 1
  fi

  TRANSFORM_DIALECT_SOURCE_FILE=$1; shift
  TRANSFORM_DIALECT_TRANSFORM_FILE=$1; shift
  FUNCTION_NAME=$1; shift
  SZ1=$1; shift
  SZ2=$1; shift

  NUM_ITERATIONS=6
  NVPROF_TRACE_FILE=/tmp/nvprof_trace
  NUM_ELEMENTS="${SZ1}*${SZ2}"
  FUNCTION_INPUT="--function_input=\"${SZ1}x${SZ2}xf32=1\""
  
  
  echo ==========================================================
  echo Reproduction instructions:
  echo iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} -- --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} \| \\
  echo   nvprof --print-gpu-trace iree-run-module --entry_function=${FUNCTION_NAME} --device=cuda ${FUNCTION_INPUT} \| \\
  echo grep ${FUNCTION_NAME}
  echo ==========================================================
  
  iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} \
      -- --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} | \
  nvprof --print-gpu-trace iree-run-module \
      --entry_function=${FUNCTION_NAME} --device=cuda ${FUNCTION_INPUT} 2>&1 | \
  grep ${FUNCTION_NAME} > ${NVPROF_TRACE_FILE}
  
  NVPROF_TRACE=$(cat ${NVPROF_TRACE_FILE})
  echo "${NVPROF_TRACE}"
  P50_TIME=$(get-p50-from-nvprof "${NVPROF_TRACE}" ${FUNCTION_NAME} ${NUM_ITERATIONS})
  ELT_PER_S=$(echo "${NUM_ELEMENTS}/${P50_TIME}" | bc -l) 
  echo ${FUNCTION_NAME} ${FUNCTION_INPUT} P50: ${P50_TIME} ns ${ELT_PER_S} GElements/s
}
