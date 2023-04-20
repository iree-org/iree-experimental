# Instructions
# ============
# ```
#    make clean && make -j 8 some_tests
# ```
# This will generate the following files:
#   - logs/*.log the runtimes for various matmul sizes
#   - gen/* various generated files with IR and PTX
#
# To produce PTX:
# ```
#   # IREE without transform dialect.
#   make gen/iree_cuda_fill_matmul_200_300_400.vmfb
#   # IREE with transform dialect.
#   make gen/iree_cuda_fill_matmul_200_300_400.td.vmfb
# ```
#
# To produce the TD script:
# ```
#   make gen/fill_matmul_2052_2052_2052.embedded.td.mlir
# ```
#
# Then further process with:
# ```
#   iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' gen/fill_matmul_2052_2052_2052.embedded.td.mlir
# ```
#
# Or: 
# ```
#   iree-compile --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 gen/fill_matmul_2052_2052_2052.embedded.td.mlir
# ```

LLVM_SRC=$(HOME)/github/llvm-project
LLVM_BUILD=$(HOME)/github/llvm-project/build
IREE_SRC=$(HOME)/github/iree
IREE_BUILD=$(HOME)/github/iree/build
IREE_SAMPLES_DIR=$(HOME)/github/iree-samples/transform_dialect/examples

MLIR_OPT=$(LLVM_BUILD)/bin/mlir-opt
IREE_OPT=$(IREE_BUILD)/tools/iree-opt
IREE_COMPILE=$(IREE_BUILD)/tools/iree-compile
IREE_RUN_MODULE=$(IREE_BUILD)/tools/iree-run-module

MATMUL_STUB=$(IREE_SAMPLES_DIR)/matmul.mlir


define extract_size_from_matmul
  $(subst _, ,$(shell echo $1 | sed -e "s/.*_matmul_//g" | sed -e "s/\..*//g" ))
endef


gen/tmp_fill_matmul_%.mlir: $(MATMUL_STUB)
	$(eval M:=$(word 1, $(call extract_size_from_matmul,$@)))
	$(eval N:=$(word 2, $(call extract_size_from_matmul,$@)))
	$(eval K:=$(word 3, $(call extract_size_from_matmul,$@)))
	@cat $(MATMUL_STUB) | sed "s/\$${M}/$M/g" | sed "s/\$${N}/$N/g" | sed "s/\$${K}/$K/g" > $@;


gen/fill_matmul_%.embedded.td.mlir: gen/fill_matmul_%.mlir $(IREE_OPT)
	@echo make $@
	@$(IREE_OPT) $< \
	  --iree-hal-target-backends=cuda \
		--iree-hal-cuda-llvm-target-arch=sm_80 \
		--iree-abi-transformation-pipeline \
		--iree-flow-transformation-pipeline \
		--iree-stream-transformation-pipeline \
		--iree-hal-configuration-pipeline | \
	$(IREE_OPT) --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" \
	  --iree-hal-target-backends=cuda \
		--iree-hal-cuda-llvm-target-arch=sm_80 \
		--iree-codegen-llvmgpu-enable-transform-dialect-jit \
		--iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy \
	> $@


gen/fill_matmul_%.mlir: gen/tmp_fill_matmul_%.mlir $(MLIR_OPT)
	@echo make $@
	@cat $< | sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
    $(MLIR_OPT) -symbol-dce > $@


gen/iree_cuda_fill_matmul_%.vmfb: gen/fill_matmul_%.mlir $(IREE_COMPILE)
	@echo make $@
	@$(IREE_COMPILE) $< --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
	  --iree-hal-benchmark-dispatch-repeat-count=5  --iree-codegen-llvmgpu-use-mma-sync \
		-o $@


gen/iree_cuda_fill_matmul_%.td.vmfb: gen/fill_matmul_%.mlir gen/fill_matmul_%.embedded.td.mlir $(IREE_COMPILE)
	@echo make $@
	@$(IREE_COMPILE) $< --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
	  --iree-hal-benchmark-dispatch-repeat-count=5 \
		--iree-codegen-llvmgpu-use-mma-sync \
		--iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy \
		-o $@


run_%: gen/%.vmfb
	@echo ssh into ${A100_MACHINE_IP} and run $<
	scp $< $(USER)@$(A100_MACHINE_IP):~/ > /dev/null 2>&1
	$(eval M:=$(word 1, $(call extract_size_from_matmul,$@)))
	$(eval N:=$(word 2, $(call extract_size_from_matmul,$@)))
	$(eval K:=$(word 3, $(call extract_size_from_matmul,$@)))
	@printf 'make %-50s -> ' "$@" > logs/$@.log
	@{ \
		ssh ${USER}@${A100_MACHINE_IP} "/usr/local/cuda/bin/nsys profile --stats=true ~/iree-run-module --function=fill_matmul_static --device=cuda  --module=$(notdir $<) --input=$(M)x$(K)xf32=1 --input=$(K)x$(N)xf32=1 --input=$(M)x$(N)xf32=1 2>&1" | \
		grep fill_matmul_static | grep -v EXEC | \
		awk '{print $$6}'; \
	} >> logs/$@.log


manual_run_%: gen/%.vmfb
	@echo ssh into ${A100_MACHINE_IP} and run $<
	scp $< $(USER)@$(A100_MACHINE_IP):~/ > /dev/null 2>&1
	$(eval M:=$(word 1, $(call extract_size_from_matmul,$@)))
	$(eval N:=$(word 2, $(call extract_size_from_matmul,$@)))
	$(eval K:=$(word 3, $(call extract_size_from_matmul,$@)))
	@{ \
		ssh ${USER}@${A100_MACHINE_IP} "/usr/local/cuda/bin/nsys profile --stats=true ~/iree-run-module --function=fill_matmul_static --device=cuda  --module=$(notdir $<) --input=$(M)x$(K)xf32=1 --input=$(K)x$(N)xf32=1 --input=$(M)x$(N)xf32=1 2>&1"; \
	}
	@{ \
		ssh ${USER}@${A100_MACHINE_IP} "sudo /usr/local/cuda/bin/ncu -f --set full -o profile ~/iree-run-module --function=fill_matmul_static --device=cuda  --module=$(notdir $<) --input=$(M)x$(K)xf32=1 --input=$(K)x$(N)xf32=1 --input=$(M)x$(N)xf32=1 2>&1"; \
	}
	@scp  ${USER}@${A100_MACHINE_IP}:~/profile.ncu-rep ~/


fill_matmul_%: run_iree_cuda_fill_matmul_% run_iree_cuda_fill_matmul_%.td
	$(NOECHO) $(NOOP)


# fill_matmul_3455_1023_2047 fill_matmul_3452_1000_1000 fill_matmul_2052_2052_2052 fill_matmul_1000_1000_3052
some_tests: gen_dir logs_dir fill_matmul_3452_1020_2044
	ls logs/*.log | sort -n | xargs cat


logs_dir:
	@mkdir -p logs


gen_dir:
	@mkdir -p gen


clean:
	rm -f logs/* gen/*


all: some_tests
	

# Keep all intermediate files for now
.SECONDARY: 


