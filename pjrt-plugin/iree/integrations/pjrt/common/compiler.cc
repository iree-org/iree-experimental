// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: Fix me.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "iree/integrations/pjrt/common/compiler.h"

#include <sys/mman.h>
#include <unistd.h>

#include <functional>
#include <iostream>  // TODO: Remove
#include <vector>

#include "iree/compiler/API2/Stub/Loader.h"

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// In-process stub compiler
//===----------------------------------------------------------------------===//

namespace {

class MMapCompilerOutput : public CompilerOutput {
 public:
  MMapCompilerOutput(void* data, size_t length)
      : data_(data), length_(length) {}
  ~MMapCompilerOutput() { munmap(data_, length_); }
  void* GetData() { return data_; }
  size_t GetDataSize() { return length_; }

 private:
  void* data_;
  size_t length_;
};

using SessionRecycler = std::function<void(iree_compiler_session_t*)>;
class InprocessCompilerJob : public CompilerJob {
 public:
  // Takes ownership of both |session| and |inv|. On destruction, destroys
  // |inv| and passes |session| to the recycler (this can be used to implement
  // session pooling).
  InprocessCompilerJob(iree_compiler_session_t* session,
                       iree_compiler_invocation_t* inv,
                       SessionRecycler session_recycler)
      : session_(session), inv_(inv), session_recycler_(session_recycler) {}
  ~InprocessCompilerJob() {
    if (error_) {
      ireeCompilerErrorDestroy(error_);
    }
    for (auto* source : retained_sources_) {
      ireeCompilerSourceDestroy(source);
    }
    ireeCompilerInvocationDestroy(inv_);
    session_recycler_(session_);

    if (output_) {
      ireeCompilerOutputDestroy(output_);
    }
  }

  std::string GetErrorMessage() override {
    if (!error_) return std::string();
    const char* cstr = ireeCompilerErrorGetMessage(error_);
    return std::string(cstr);
  }

  bool SetFlag(const char* flag) {
    auto* error = ireeCompilerSessionSetFlags(session_, 1, &flag);
    if (error) {
      SetError(error);
      return false;
    }
    return true;
  }

  bool ParseSourceBuffer(const void* buffer, size_t length) override {
    iree_compiler_source_t* source;
    auto* error = ireeCompilerSourceWrapBuffer(
        session_, "<jit>", static_cast<const char*>(buffer), length, &source);
    if (error) {
      SetError(error);
      return false;
    }
    retained_sources_.push_back(source);

    return ireeCompilerInvocationParseSource(inv_, source);
  }

  std::unique_ptr<CompilerOutput> CompileStandardPipeline() override {
    iree_compiler_error_t* error;
    if (!ireeCompilerInvocationPipeline(inv_, IREE_COMPILER_PIPELINE_STD)) {
      return nullptr;
    }

    // Setup temp file output.
    output_fd_ = memfd_create("output.vmfb", 0);
    if (output_fd_ == -1) {
      // TODO: Better error handling.
      return nullptr;
    }
    error = ireeCompilerOutputOpenFD(output_fd_, &output_);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Output.
    error = ireeCompilerInvocationOutputVMBytecode(inv_, output_);
    if (error) {
      SetError(error);
      return nullptr;
    }

    // Map the data.
    off_t fsize;
    fsize = lseek(output_fd_, 0, SEEK_END);
    void* output_data =
        mmap(nullptr, fsize, PROT_READ | PROT_EXEC, MAP_SHARED, output_fd_, 0);

    return std::make_unique<MMapCompilerOutput>(output_data, fsize);
  }

 private:
  void SetError(iree_compiler_error_t* error) {
    if (error_) {
      ireeCompilerErrorDestroy(error_);
    }
    error_ = error;
  }

  iree_compiler_session_t* session_;
  iree_compiler_invocation_t* inv_;
  SessionRecycler session_recycler_;

  std::vector<iree_compiler_source_t*> retained_sources_;
  iree_compiler_error_t* error_ = nullptr;

  // Output.
  iree_compiler_output_t* output_ = nullptr;
  int output_fd_ = -1;  // Owned by output_.
};

}  // namespace

std::shared_ptr<AbstractCompiler> InprocessStubCompiler::Initialize(
    const char* libraryPath) {
  if (!ireeCompilerLoadLibrary(libraryPath)) {
    return nullptr;
  }

  ireeCompilerGlobalInitialize(/*initializeCommandLine=*/false);
  return std::make_shared<InprocessStubCompiler>();
}

std::unique_ptr<CompilerJob> InprocessStubCompiler::StartJob() {
  auto* session = ireeCompilerSessionCreate();
  auto* inv = ireeCompilerInvocationCreate(session);

  // TODO: Capture diagnostics, etc vs spewing to stderr.
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);

  return std::make_unique<InprocessCompilerJob>(
      session, inv, [](iree_compiler_session_t* session) {
        ireeCompilerSessionDestroy(session);
      });
}

}  // namespace iree::pjrt
