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

  # echo iree-transform-opt-dispatch-only ${MLIR_FILE} -b ${BACKEND} -c ${CODEGEN_SPEC_FILE} -d ${DISPATCH_SPEC_FILE} \| \\
  # echo iree-opt --iree-hal-target-backends=${BACKEND} --iree-stream-transformation-pipeline --iree-hal-configuration-pipeline \| \\
  # echo iree-opt ${CODEGEN_FLAG} ${EXTRA_ARGS}

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
  if [[ "$#" -lt  3 ]]; then
    echo "Usage: get-p50-from-nvprof nvprof_trace_string function_name num_iterations"
    return 1
  fi

  NVPROF_TRACE=$1
  FUNCTION_NAME=$2
  NUM_ITERATIONS=$3


  P50_TIME=$(echo "${NVPROF_TRACE}" | \
    grep ${FUNCTION_NAME}_dispatch | \
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
function get-ops-from-elemental-type() {
  if [[ $# != 1 ]]; then 
    echo "Usage: get-ops-from-elemental-type"
    return
  fi

  ELEMENTAL_TYPE=$1
  if [[ ${ELEMENTAL_TYPE::1} == "f" ]]; then
    echo "arith.addf" "arith.divf" "-0.000000e+00"
  elif [[ ${ELEMENTAL_TYPE::1} == "i" ]]; then
    echo "arith.addi" "arith.divui" "0"
  else
    echo "Unknown element in get-ops-from-elemental-type"
  fi    
}

function benchmark-transform-create() {
  cmake --build ./build --target iree-opt > /dev/null
  if [[ $? -ne 0 ]]; then 
    echo "Compilation failed"
    return 1
  fi
  
  if [[ "$#" -lt  7 || "$#" -gt  10 ]]; then
    echo "Got "$#" args"
    echo "Usage: benchmark-transform-create [-r] -b <backend> stub_file_name function_name ELEMENTAL_TYPE SZ1 [SZ2] [SZ3] [SZ4] "
    return 1
  fi

  RUN=0
  if [[ $1 == "-r" ]]; then
    RUN=1; shift
  else
    : # noop
  fi
  if [[ $1 == "-b" ]]; then
    shift
    BACKEND=$1; shift
  else
    echo "Expected -b <backend> but got "$1
    echo "Usage: benchmark-transform-create [-r] -b <backend> stub_file_name function_name ELEMENTAL_TYPE SZ1 [SZ2] [SZ3] [SZ4] "
    return 1
  fi
  if [[ ${BACKEND} != "cuda" && ${BACKEND} != "llvm-cpu" ]]; then
    echo "Unknown IREE backend: " ${BACKEND}
    return 1
  fi
  TRANSFORM_DIALECT_BENCHMARK_STUB_FILE=$1; shift
  FUNCTION_NAME=$1; shift
  ELEMENTAL_TYPE=$1; shift

  SIZES_LIST=$@
  IFS=' ' read -r -a SIZES <<< "${SIZES_LIST}"
  FILE_NAME_SIZES=$(echo ${SIZES[@]} | sed "s/ /x/g")
  FUNCTION_INPUT=$(echo ${SIZES[@]} | sed "s/ /x/g" | sed "s/x0//g")
  FUNCTION_INPUT="--function_input=\"${FUNCTION_INPUT}x${ELEMENTAL_TYPE}=1\""

  read -r ADD_OP DIV_OP ZERO <<< $(get-ops-from-elemental-type ${ELEMENTAL_TYPE})

  NUM_ITERATIONS=6

  TRANSFORM_DIALECT_TMP_SOURCE_FILE=/tmp/tmp_${FUNCTION_NAME}_${FILE_NAME_SIZES}.mlir
  TRANSFORM_DIALECT_SOURCE_FILE=/tmp/${FUNCTION_NAME}_${FILE_NAME_SIZES}x${ELEMENTAL_TYPE}.mlir
  # Extract exactly the func we care about and let `mlir-opt -symbol-dce ` clean
  # up the rest of the IR.
  # This lets us use files with multiple funcs
  cat ${TRANSFORM_DIALECT_BENCHMARK_STUB_FILE} > ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  sed -i "s/private @${FUNCTION_NAME}(/@${FUNCTION_NAME}(/g" ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  sed -i "s/\${ELEMENTAL_TYPE}/${ELEMENTAL_TYPE}/g" ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  sed -i "s/\${ZERO}/${ZERO}/g" ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  sed -i "s/\${ADD_OP}/${ADD_OP}/g" ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  sed -i "s/\${DIV_OP}/${DIV_OP}/g" ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  sed -i "s/\${SZ1}/$(test ${SIZES[0]} && echo ${SIZES[0]} || echo 0 )/g" ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  sed -i "s/\${SZ2}/$(test ${SIZES[1]} && echo ${SIZES[1]} || echo 0 )/g" ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  sed -i "s/\${SZ3}/$(test ${SIZES[2]} && echo ${SIZES[2]} || echo 0 )/g" ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  sed -i "s/\${SZ4}/$(test ${SIZES[3]} && echo ${SIZES[3]} || echo 0 )/g" ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
  
  # echo mlir-opt ${TRANSFORM_DIALECT_TMP_SOURCE_FILE} -symbol-dce > ${TRANSFORM_DIALECT_SOURCE_FILE}
  mlir-opt ${TRANSFORM_DIALECT_TMP_SOURCE_FILE} -symbol-dce > ${TRANSFORM_DIALECT_SOURCE_FILE}

  echo DUMP_TRANSFORM_REPRO=1 iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b ${BACKEND}  -- --mlir-disable-threading
  TRANSFORM_DIALECT_TRANSFORM_FILE=$(DUMP_TRANSFORM_REPRO=1 iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b ${BACKEND}  -- --mlir-disable-threading 2>&1 > /dev/null | grep iree-opt | awk '{print $4}')
  
  if [[ ${TRANSFORM_DIALECT_TRANSFORM_FILE} =~ "tmp" ]] ; then
    : # noop
  else
    echo "Could not create TRANSFORM_DIALECT_TRANSFORM_FILE (match failure or compile failure)?"
    echo Try inspecting: ${TRANSFORM_DIALECT_TMP_SOURCE_FILE}
    echo Try running:   DUMP_TRANSFORM_REPRO=1 iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b ${BACKEND}  -- --mlir-disable-threading
    return 1
  fi

  if [ -z ${TRANSFORM_DIALECT_NO_DEBUG+x} ]; then
    echo ==========================================================
    echo Problem created successfully, reproduction instructions:
    echo ==========================================================
    echo Transform dialect source file is: 
    echo "    "${TRANSFORM_DIALECT_SOURCE_FILE}
    echo Transform dialect transform file is: 
    echo "    "${TRANSFORM_DIALECT_TRANSFORM_FILE}
    echo Dump transformed IR with: 
    echo "    "benchmark-transform-run-iree-opt -b ${BACKEND} ${TRANSFORM_DIALECT_SOURCE_FILE} ${TRANSFORM_DIALECT_TRANSFORM_FILE}
    echo Dump transformed binary/PTX with: 
    echo "    "benchmark-transform-run-iree-compile -b ${BACKEND} ${TRANSFORM_DIALECT_SOURCE_FILE} ${TRANSFORM_DIALECT_TRANSFORM_FILE}
    if [[ ${BACKEND} == "cuda" ]]; then
      echo Run nvprof with e.g.: 
      echo "    "benchmark-transform-run-nvprof ${TRANSFORM_DIALECT_SOURCE_FILE} ${TRANSFORM_DIALECT_TRANSFORM_FILE} ${FUNCTION_NAME} ${ELEMENTAL_TYPE} ${SIZES_LIST}
      echo Run nvprof without transform dialect: 
      echo "    "iree-compile ${TRANSFORM_DIALECT_SOURCE_FILE} --iree-hal-target-backends=${BACKEND} --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} \| \\
      echo "    "nvprof --print-gpu-trace iree-run-module --entry_function=${FUNCTION_NAME} --device=${BACKEND} ${FUNCTION_INPUT} 2\>\&1 \| \\
      echo "    "grep ${FUNCTION_NAME}_dispatch
    else
      # Non-GPU run not yet supported
      : #noop
    fi
    echo ==========================================================
  fi

  # echo iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b ${BACKEND}  -- --mlir-disable-threading 2>&1 > /dev/null || exit 1
  iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b ${BACKEND}  -- --mlir-disable-threading 2>&1 > /dev/null || exit 1

  if [[ ${RUN} != 0 ]]; then
    if [[ ${BACKEND} == "cuda" ]]; then
      benchmark-transform-run-nvprof ${TRANSFORM_DIALECT_SOURCE_FILE} ${TRANSFORM_DIALECT_TRANSFORM_FILE} ${FUNCTION_NAME} ${ELEMENTAL_TYPE} ${SIZES_LIST}
    else
      benchmark-transform-run-cpu ${TRANSFORM_DIALECT_SOURCE_FILE} ${TRANSFORM_DIALECT_TRANSFORM_FILE} ${FUNCTION_NAME} ${ELEMENTAL_TYPE} ${SIZES_LIST}
    fi
  fi
}

function benchmark-transform-run-iree-opt() {
  cmake --build ./build --target iree-opt > /dev/null
  if [[ $? -ne 0 ]]; then 
    echo "Compilation failed"
    return 1
  fi

  if [[ "$#" -lt  4 ]]; then
    echo "Usage: benchmark-transform-run-iree-opt -b <backend> source-file transform-file [optional extra args]"
    return 1
  fi

  if [[ $1 == "-b" ]]; then
    shift
    BACKEND=$1; shift
  else
    echo "Usage: benchmark-transform-run-iree-opt -b <backend> source-file transform-file [optional extra args]"
    return 1
  fi
  if [[ ${BACKEND} != "cuda" && ${BACKEND} != "llvm-cpu" ]]; then
    echo "Unknown IREE backend: " ${BACKEND}
    return 1
  fi
  TRANSFORM_DIALECT_SOURCE_FILE=$1; shift
  TRANSFORM_DIALECT_TRANSFORM_FILE=$1; shift
  echo iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b ${BACKEND} -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} -- $@
  iree-transform-opt ${TRANSFORM_DIALECT_SOURCE_FILE} -b ${BACKEND} -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} -- $@
}

function benchmark-transform-run-iree-compile() {
  cmake --build ./build --target iree-compile > /dev/null
  if [[ $? -ne 0 ]]; then 
    echo "Compilation failed"
    return 1
  fi

  if [[ "$#" -lt  4 ]]; then
    echo "Usage: benchmark-transform-run-iree-compile -b <backend> source-file transform-file [optional extra args]"
    return 1
  fi

  if [[ $1 == "-b" ]]; then
    shift
    BACKEND=$1; shift
  else
    echo "Usage: benchmark-transform-run-iree-compile -b <backend> source-file transform-file [optional extra args]"
    return 1
  fi
  if [[ ${BACKEND} != "cuda" && ${BACKEND} != "llvm-cpu" ]]; then
    echo "Unknown IREE backend: " ${BACKEND}
    return 1
  fi
  TRANSFORM_DIALECT_SOURCE_FILE=$1; shift
  TRANSFORM_DIALECT_TRANSFORM_FILE=$1; shift
  echo iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b ${BACKEND} -c ${TRANSFORM_DIALECT_TRANSFORM_FILE}
  iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b ${BACKEND} -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} -- $@
}

function benchmark-transform-run-cpu() {
  cmake --build ./build --target iree-compile iree-run-module iree-benchmark-module > /dev/null
  if [[ $? -ne 0 ]]; then 
    echo "Compilation failed"
    return 1
  fi

  NUM_ITERATIONS=10

  if [[ "$#" -lt  5 || "$#" -gt  7 ]]; then
    echo "Usage: benchmark-transform-run-cpu source-file transform-file function_name ELEMENTAL_TYPE SZ1 [SZ2] [SZ3] [SZ4]"
    return 1
  fi

  TRANSFORM_DIALECT_SOURCE_FILE=$1; shift
  TRANSFORM_DIALECT_TRANSFORM_FILE=$1; shift
  FUNCTION_NAME=$1; shift
  ELEMENTAL_TYPE=$1; shift

  SIZES_LIST=$@
  IFS=' ' read -r -a SIZES <<< "${SIZES_LIST}"
  NUM_ELEMENTS=$(echo ${SIZES[@]} | sed "s/ /*/g" | sed "s/0/1/g")
  FILE_NAME_SIZES=$(echo ${SIZES[@]} | sed "s/ /x/g")
  FUNCTION_INPUT=$(echo ${SIZES[@]} | sed "s/ /x/g" | sed "s/x0//g")
  FUNCTION_INPUT="--function_input=\"${FUNCTION_INPUT}x${ELEMENTAL_TYPE}=1\""

  if [ -z ${TRANSFORM_DIALECT_NO_DEBUG+x} ]; then
    echo ==========================================================
    echo Reproduction instructions:
    echo -- With transform dialect:
    echo "    "iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b llvm-cpu -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} \\
    echo "    "-- --iree-llvm-target-triple=x86_64-pc-linux-gnu \\
    echo "    "   --iree-llvm-target-cpu-features=host \\
    echo "    "   --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} \| \\
    echo "    "iree-benchmark-module --entry_function=${FUNCTION_NAME} --device=local-task --task_topology_group_count=1 --batch_size=${NUM_ITERATIONS} ${FUNCTION_INPUT}
    echo -- Without transform dialect:
    echo "    "iree-compile ${TRANSFORM_DIALECT_SOURCE_FILE} --iree-hal-target-backends=llvm-cpu \\
    echo "    "   --iree-llvm-target-triple=x86_64-pc-linux-gnu \\
    echo "    "   --iree-llvm-target-cpu-features=host \\
    echo "    "   --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} \| \\
    echo "    "iree-benchmark-module --entry_function=${FUNCTION_NAME} --device=local-task --task_topology_group_count=1 --batch_size=${NUM_ITERATIONS} ${FUNCTION_INPUT}
    echo ==========================================================
  fi
  
  ###
  ### With transform dialect:
  ###
  iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b llvm-cpu -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} \
      -- --iree-llvm-target-triple=x86_64-pc-linux-gnu \
         --iree-llvm-target-cpu-features=host \
         --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} | \
  iree-benchmark-module --entry_function=${FUNCTION_NAME} --device=local-task --task_topology_group_count=1 --batch_size=${NUM_ITERATIONS} ${FUNCTION_INPUT}


  # iree-benchmark-module --entry_function=${FUNCTION_NAME} --device=cuda ${FUNCTION_INPUT} 2>&1  | \
  # grep ${FUNCTION_NAME}_dispatch > ${NVPROF_TRACE_FILE}
  
  # NVPROF_TRACE=$(cat ${NVPROF_TRACE_FILE})
  # if [ -z ${TRANSFORM_DIALECT_NO_DEBUG+x} ]; then
  #   echo "${NVPROF_TRACE}"
  # fi
  # P50_TIME=$(get-p50-from-nvprof "${NVPROF_TRACE}" ${FUNCTION_NAME} ${NUM_ITERATIONS})
  # ELT_PER_S=$(echo "${NUM_ELEMENTS}/${P50_TIME}" | bc -l) 
  # echo With transform dialect: ${FUNCTION_NAME} ${FUNCTION_INPUT} P50: ${P50_TIME} ns ${ELT_PER_S} GElements/s

  # ###
  # ### Without transform dialect:
  # ###
  # iree-compile ${TRANSFORM_DIALECT_SOURCE_FILE} --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} | \
  # nvprof --print-gpu-trace iree-run-module --entry_function=${FUNCTION_NAME} --device=cuda ${FUNCTION_INPUT} 2>&1 | \
  # grep ${FUNCTION_NAME}_dispatch > ${NVPROF_TRACE_FILE}
  
  # NVPROF_TRACE=$(cat ${NVPROF_TRACE_FILE})
  # if [ -z ${TRANSFORM_DIALECT_NO_DEBUG+x} ]; then
  #   echo "${NVPROF_TRACE}"
  # fi
  # P50_TIME=$(get-p50-from-nvprof "${NVPROF_TRACE}" ${FUNCTION_NAME} ${NUM_ITERATIONS})
  # ELT_PER_S=$(echo "${NUM_ELEMENTS}/${P50_TIME}" | bc -l) 
  # echo Without transform dialect: ${FUNCTION_NAME} ${FUNCTION_INPUT} P50: ${P50_TIME} ns ${ELT_PER_S} GElements/s
}

function benchmark-transform-run-nvprof() {
  cmake --build ./build --target iree-compile iree-run-module > /dev/null
  if [[ $? -ne 0 ]]; then 
    echo "Compilation failed"
    return 1
  fi

  NUM_ITERATIONS=10
  NVPROF_TRACE_FILE=/tmp/nvprof_trace

  if [[ "$#" -lt  5 || "$#" -gt  7 ]]; then
    echo "Usage: benchmark-transform-run-nvprof source-file transform-file function_name ELEMENTAL_TYPE SZ1 [SZ2] [SZ3] [SZ4]"
    return 1
  fi

  TRANSFORM_DIALECT_SOURCE_FILE=$1; shift
  TRANSFORM_DIALECT_TRANSFORM_FILE=$1; shift
  FUNCTION_NAME=$1; shift
  ELEMENTAL_TYPE=$1; shift

  SIZES_LIST=$@
  IFS=' ' read -r -a SIZES <<< "${SIZES_LIST}"
  NUM_ELEMENTS=$(echo ${SIZES[@]} | sed "s/ /*/g" | sed "s/0/1/g")
  FILE_NAME_SIZES=$(echo ${SIZES[@]} | sed "s/ /x/g")
  FUNCTION_INPUT=$(echo ${SIZES[@]} | sed "s/ /x/g" | sed "s/x0//g")
  FUNCTION_INPUT="--function_input=\"${FUNCTION_INPUT}x${ELEMENTAL_TYPE}=1\""

  if [ -z ${TRANSFORM_DIALECT_NO_DEBUG+x} ]; then
    echo ==========================================================
    echo Reproduction instructions:
    echo -- With transform dialect
    echo "    "iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} -- --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} \| \\
    echo "    "nvprof --print-gpu-trace iree-run-module --entry_function=${FUNCTION_NAME} --device=cuda ${FUNCTION_INPUT} 2\>\&1 \| \\
    echo "    "grep ${FUNCTION_NAME}_dispatch
    echo -- Without transform dialect: 
    echo "    "iree-compile ${TRANSFORM_DIALECT_SOURCE_FILE} --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} \| \\
    echo "    "nvprof --print-gpu-trace iree-run-module --entry_function=${FUNCTION_NAME} --device=cuda ${FUNCTION_INPUT} 2\>\&1 \| \\
    echo "    "grep ${FUNCTION_NAME}_dispatch
    echo ==========================================================
  fi
  
  ###
  ### With transform dialect:
  ###
  iree-transform-compile ${TRANSFORM_DIALECT_SOURCE_FILE} -b cuda -c ${TRANSFORM_DIALECT_TRANSFORM_FILE} \
      -- --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} | \
  nvprof --print-gpu-trace iree-run-module --entry_function=${FUNCTION_NAME} --device=cuda ${FUNCTION_INPUT} 2>&1  | \
  grep ${FUNCTION_NAME}_dispatch > ${NVPROF_TRACE_FILE}
  
  NVPROF_TRACE=$(cat ${NVPROF_TRACE_FILE})
  if [ -z ${TRANSFORM_DIALECT_NO_DEBUG+x} ]; then
    echo "${NVPROF_TRACE}"
  fi
  P50_TIME=$(get-p50-from-nvprof "${NVPROF_TRACE}" ${FUNCTION_NAME} ${NUM_ITERATIONS})
  ELT_PER_S=$(echo "${NUM_ELEMENTS}/${P50_TIME}" | bc -l) 
  echo With transform dialect: ${FUNCTION_NAME} ${FUNCTION_INPUT} P50: ${P50_TIME} ns ${ELT_PER_S} GElements/s

  ###
  ### Without transform dialect:
  ###
  iree-compile ${TRANSFORM_DIALECT_SOURCE_FILE} --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=${NUM_ITERATIONS} | \
  nvprof --print-gpu-trace iree-run-module --entry_function=${FUNCTION_NAME} --device=cuda ${FUNCTION_INPUT} 2>&1 | \
  grep ${FUNCTION_NAME}_dispatch > ${NVPROF_TRACE_FILE}
  
  NVPROF_TRACE=$(cat ${NVPROF_TRACE_FILE})
  if [ -z ${TRANSFORM_DIALECT_NO_DEBUG+x} ]; then
    echo "${NVPROF_TRACE}"
  fi
  P50_TIME=$(get-p50-from-nvprof "${NVPROF_TRACE}" ${FUNCTION_NAME} ${NUM_ITERATIONS})
  ELT_PER_S=$(echo "${NUM_ELEMENTS}/${P50_TIME}" | bc -l) 
  echo Without transform dialect: ${FUNCTION_NAME} ${FUNCTION_INPUT} P50: ${P50_TIME} ns ${ELT_PER_S} GElements/s
}

function test-benchmark-transform-create() {
  if [[ $1 == "-b" ]]; then
    shift
    BACKEND=$1; shift
  else
    echo "Usage: test-benchmark-transform-create -b <backend> stub-file"
    return 1
  fi
  if [[ ${BACKEND} != "cuda" && ${BACKEND} != "llvm-cpu" ]]; then
    echo "Unknown IREE backend: " ${BACKEND}
    return 1
  fi

  STUB_FILENAME=$1
  benchmark-transform-create -r -b ${BACKEND} ${STUB_FILENAME} reduction_2d_static f16 2 3 && \
  benchmark-transform-create -r -b ${BACKEND} ${STUB_FILENAME} reduction_2d_elementwise_static f32 2 3 && \
  benchmark-transform-create -r -b ${BACKEND} ${STUB_FILENAME} reduction_3d_elementwise_static f16 2 3 4 && \
  benchmark-transform-create -r -b ${BACKEND} ${STUB_FILENAME} reduction_2d_dynamic f32 2 3 && \
  benchmark-transform-create -r -b ${BACKEND} ${STUB_FILENAME} reduction_2d_elementwise_dynamic f64 2 3

  # TODO: the vanilla 3-d case is collapsed by IREE and we fail to match it.
  # benchmark-transform-create -r -b cuda ${STUB_FILENAME} reduction_3d_static 2 3 4
}
