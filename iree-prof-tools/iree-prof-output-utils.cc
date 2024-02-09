// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-prof-tools/iree-prof-output-utils.h"

#include "third_party/abseil-cpp/absl/base/log_severity.h"
#include "third_party/abseil-cpp/absl/flags/parse.h"
#include "third_party/abseil-cpp/absl/log/globals.h"
#include "third_party/abseil-cpp/absl/log/initialize.h"
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/abseil-cpp/absl/strings/string_view.h"
#include "third_party/abseil-cpp/absl/time/clock.h"
#include "third_party/abseil-cpp/absl/time/time.h"
#include "third_party/tracy/server/TracyWorker.hpp"

namespace iree_prof {
namespace {

constexpr int kGpuThreadIndicator = 1 << 16;

// Sum of durations of zones in a timeline. A timeline of a zone consists of
// sub-zones enclosed within the given zone, e.g. a function calling functions.
// A thread has a timeline with top-level zones, i.e. ones without parent zones.
template <typename T>
int64_t SumDurationInTimeline(const tracy::Worker& worker,
                              const tracy::Vector<T>& timeline) {
  int64_t duration = 0;
  for (const auto& e : timeline) {
    const auto& event = GetEvent(e);
    duration += GetEventDuration(event);
  }
  return duration;
}

// Decompresses the thread ID associated to a GPU zone if it looks like a
// compressed thread ID. If not, looks for the thread ID from GPU contexts.
//
// Each GPU zone is associated to a CPU thread issuing it. When tracy is
// collecting live events, a valid compressed thread ID is set to GPU zones.
// When tracy writes events to a file, it writes the uncompressed thread id
// (int64_t) inproperly as a compressed thread id (uint16_t). So, when a GPU
// zone is rebuilt from a tracy file, GPU thread information is wrong.
// This function looks through GPU contexts and matches last 16bits, i.e. the
// size of uint16_t to figure out the original thread ID.
uint64_t DecompressOrFixGpuThreadId(const tracy::Worker& worker,
                                    uint16_t gpu_thread_id) {
  if (gpu_thread_id < worker.GetThreadData().size()) {
    return worker.DecompressThread(gpu_thread_id);
  }

  for (const auto& t : worker.GetThreadData()) {
    if (gpu_thread_id == static_cast<uint16_t>(t->id)) {
      return t->id;
    }
  }

  return gpu_thread_id;
}

}  // namespace

int64_t GetEventStart(const tracy::ZoneEvent& event) {
  return event.Start();
}

int64_t GetEventStart(const tracy::GpuEvent& event) {
  return event.GpuStart();
}

int64_t GetEventEnd(const tracy::ZoneEvent& event) {
  return event.End();
}

int64_t GetEventEnd(const tracy::GpuEvent& event) {
  return event.GpuEnd();
}

const tracy::Vector<tracy::short_ptr<tracy::ZoneEvent>>* GetEventChildren(
    const tracy::Worker& worker, const tracy::ZoneEvent& event) {
  return event.HasChildren() ? &worker.GetZoneChildren(event.Child()) : nullptr;
}

const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>* GetEventChildren(
    const tracy::Worker& worker, const tracy::GpuEvent& event) {
  return event.Child() >= 0 ? &worker.GetGpuChildren(event.Child()) : nullptr;
}

int GetThreadId(const tracy::Worker::ZoneThreadData& t) {
  return t.Thread();
}

int GetThreadId(const tracy::Worker::GpuZoneThreadData& t) {
  return kGpuThreadIndicator + t.Thread();
}

template <>
std::string GetThreadName<tracy::ZoneEvent>(const tracy::Worker& worker,
                                            int thread_id) {
  return worker.GetThreadName(worker.DecompressThread(thread_id));
}

template <>
std::string GetThreadName<tracy::GpuEvent>(const tracy::Worker& worker,
                                           int thread_id) {
  uint16_t original_id = thread_id - kGpuThreadIndicator;
  int fixed_id = DecompressOrFixGpuThreadId(worker, original_id);
  for (const auto& d : worker.GetGpuData()) {
    for (const auto& t : d->threadData) {
      if (t.first == fixed_id) {
        return absl::StrCat(worker.GetString(d->name), "-", fixed_id);
      }
    }
  }
  return absl::StrCat("gpu-thread-", fixed_id);
}

template <>
std::string GetThreadName<tracy::Worker::SourceLocationZones>(
    const tracy::Worker& worker, int thread_id) {
  return GetThreadName<tracy::ZoneEvent>(worker, thread_id);
}

template <>
std::string GetThreadName<tracy::Worker::GpuSourceLocationZones>(
    const tracy::Worker& worker, int thread_id) {
  return GetThreadName<tracy::GpuEvent>(worker, thread_id);
}

template <>
int64_t GetThreadDuration<tracy::Worker::SourceLocationZones>(
    const tracy::Worker& worker,
    int thread_id) {
  const auto* data = worker.GetThreadData(worker.DecompressThread(thread_id));
  // timeline.is_magic() is false when tracy is collecting live events, i.e.
  // storing zone events in a vector indirectly via tracy::short_ptr.
  // timeline.is_magic() is true when zone events were from tracy file, i.e.
  // stored in a vector directly.
  if (!data->timeline.is_magic()) {
    return SumDurationInTimeline(worker, data->timeline);
  }
  return SumDurationInTimeline(
             worker,
             *reinterpret_cast<const tracy::Vector<tracy::ZoneEvent>*>(
                 &data->timeline));
}

template <>
int64_t GetThreadDuration<tracy::Worker::GpuSourceLocationZones>(
    const tracy::Worker& worker,
    int thread_id) {
  uint16_t original_id = thread_id - kGpuThreadIndicator;
  int fixed_id = DecompressOrFixGpuThreadId(worker, original_id);
  for (const auto& d : worker.GetGpuData()) {
    for (const auto& t : d->threadData) {
      if (t.first == fixed_id) {
        // See GetThreadDuration<tracy::Worker::SourceLocationZones>() for the
        // comment about timeline.is_magic().
        if (!t.second.timeline.is_magic()) {
          return SumDurationInTimeline(worker, t.second.timeline);
        }
        return SumDurationInTimeline(
                   worker,
                   *reinterpret_cast<const tracy::Vector<tracy::GpuEvent>*>(
                       &t.second.timeline));
      }
    }
  }
  return 0;
}

const char* GetZoneName(const tracy::Worker& worker,
                        int16_t source_location_id) {
  return worker.GetZoneName(worker.GetSourceLocation(source_location_id));
}

std::string GetSourceFileLine(const tracy::Worker& worker,
                              int16_t source_location_id) {
  const auto& zone = worker.GetSourceLocation(source_location_id);
  absl::string_view file_name = worker.GetString(zone.file);
  if (file_name.empty() || file_name == "-") {
    return "";
  }
  return absl::StrCat(file_name, ":", zone.line);
}

const tracy::PlotData* GetMemoryPlotData(const tracy::Worker& worker) {
  for (const auto* p : worker.GetPlots()) {
    if (p->type == tracy::PlotType::Memory) {
      return p;
    }
  }
  return nullptr;
}

void YieldCpu() {
  absl::SleepFor(absl::Milliseconds(100));
}

std::vector<char*> InitializeLogAndParseCommandLine(int argc, char* argv[]) {
  // Set default logging level to stderr to INFO. It can be overridden by
  // --stderrthreshold flag.
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  return absl::ParseCommandLine(argc, argv);
}

}  // namespace iree_prof
