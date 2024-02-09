// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "iree-prof-tools/iree-prof-output.h"
#include "iree-prof-tools/iree-prof-output-utils.h"
#include "third_party/abseil-cpp/absl/flags/flag.h"
#include "third_party/abseil-cpp/absl/log/log.h"
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/tracy/server/TracyWorker.hpp"

ABSL_FLAG(int, tracy_port, 18086, "TCP port number for tracy profiler.");

namespace {

// Flag to inform main thread to disconnect from tracy profiler on SIGINT.
std::atomic<bool> g_disconnect = false;

void HandleSigInt(int sigid) {
  g_disconnect = true;
}

// Runs a subprogram with |argv| which is null-terminated at the end.
absl::Status RunExecutable(char** argv) {
  int pid = fork();
  if (pid == 0) {
    std::string tracy_port_str = absl::StrCat(absl::GetFlag(FLAGS_tracy_port));
    setenv("TRACY_NO_EXIT", "1", /*overwrite=*/1);
    setenv("TRACY_PORT", tracy_port_str.c_str(), /*overwrite=*/1);
    LOG(INFO) << "Executing " << argv[0];
    execvp(argv[0], argv);
    exit(errno);  // Not expected to be reached.
  }

  if (pid < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("Can't execute ", argv[0]));
  }

  // Wait a little bit to check if a child process is still running.
  iree_prof::YieldCpu();

  int wstatus;
  int ret = waitpid(pid, &wstatus, WNOHANG);
  if (ret == 0) {
    return absl::OkStatus();
  }

  if (ret > 0) {
    if (WIFSIGNALED(wstatus)) {
      errno = EINTR;
    } else {
      errno = WEXITSTATUS(wstatus);
    }
  }
  return absl::ErrnoToStatus(errno, absl::StrCat("Can't execute ", argv[0]));
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<char*> remain_args =
      iree_prof::InitializeLogAndParseCommandLine(argc, argv);

  std::unique_ptr<tracy::Worker> worker;
  // Note that remain_args[0] has argv[0].
  if (remain_args.size() < 2) {
    LOG(ERROR) << "Executable command or a tracy file to read is not provided.";
    return EXIT_FAILURE;
  }

  // argv[] must end with nullptr.
  remain_args.push_back(nullptr);
  // remain_args[0] is this program name, i.e. iree-prof.
  auto status = RunExecutable(&remain_args[1]);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return EXIT_FAILURE;
  }

  int tracy_port = absl::GetFlag(FLAGS_tracy_port);
  worker = std::make_unique<tracy::Worker>("127.0.0.1", tracy_port);

  while (!worker->HasData()) {
    auto status = worker->GetHandshakeStatus();
    if (status != tracy::HandshakePending &&
        status != tracy::HandshakeWelcome) {
      LOG(ERROR) << "Could not connect to " << remain_args[1];
      return EXIT_FAILURE;
    }
    iree_prof::YieldCpu();
  }

  LOG(INFO) << "Connected to " << remain_args[1]
            << " through 127.0.0.1:" << tracy_port;

  signal(SIGINT, HandleSigInt);

  while(worker->IsConnected()) {
    if (g_disconnect) {
      worker->Disconnect();
    } else {
      iree_prof::YieldCpu();
    }
  }

  iree_prof::Output(*worker);
  return EXIT_SUCCESS;
}
