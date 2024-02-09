// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PROF_OUTPUT_UTILS_H_
#define IREE_PROF_OUTPUT_UTILS_H_

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "third_party/tracy/server/TracyWorker.hpp"

namespace iree_prof {

// Function templates and overloads for CPU and GPU zones.

// Templete overloads to return T either from T or from tracy::short_ptr<T>
// which is useful to get T from tracy::Vector<>, e.g. thread timeline.
template <typename T>
const T& GetEvent(const T& event) { return event; }

template <typename T>
const T& GetEvent(const tracy::short_ptr<T>& event) {
  return *event;
}

// Template to get (GPU)ZoneThreadData or (Gpu)ZoneEvent type from
// (GPU)SourceLocaitonZones which are private type of tracy::Worker.
template <typename T>
struct ZoneTypeHelper {
  using ThreadDataType = decltype(*std::declval<T>().zones.begin());
  using EventType =
      std::remove_reference_t<decltype(*std::declval<ThreadDataType>().Zone())>;
};

int64_t GetEventStart(const tracy::ZoneEvent& event);
int64_t GetEventStart(const tracy::GpuEvent& event);

int64_t GetEventEnd(const tracy::ZoneEvent& event);
int64_t GetEventEnd(const tracy::GpuEvent& event);

template <typename T>
int64_t GetEventDuration(const T& event) {
  return GetEventEnd(event) - GetEventStart(event);
}

const tracy::Vector<tracy::short_ptr<tracy::ZoneEvent>>* GetEventChildren(
    const tracy::Worker& worker, const tracy::ZoneEvent& event);
const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>* GetEventChildren(
    const tracy::Worker& worker, const tracy::GpuEvent& event);

int GetThreadId(const tracy::Worker::ZoneThreadData& t);
int GetThreadId(const tracy::Worker::GpuZoneThreadData& t);

template <typename T>
std::string GetThreadName(const tracy::Worker& worker, int thread_id);
template <>
std::string GetThreadName<tracy::ZoneEvent>(const tracy::Worker& worker,
                                            int thread_id);
template <>
std::string GetThreadName<tracy::GpuEvent>(const tracy::Worker& worker,
                                           int thread_id);

// Returns the total duration of the thread of |thread_id|. It is the sum of
// durations of top-level zones.
template <typename T>
int64_t GetThreadDuration(const tracy::Worker& worker, int thread_id);
template <>
int64_t GetThreadDuration<tracy::ZoneEvent>(const tracy::Worker& worker,
                                            int thread_id);
template <>
int64_t GetThreadDuration<tracy::GpuEvent>(const tracy::Worker& worker,
                                           int thread_id);

// Gets source file:line string. May return an empty string if it is unknown.
std::string GetSourceFileLine(const tracy::Worker& worker,
                              int16_t source_location_id);

// Gets plot data of memory usage.
const tracy::PlotData* GetMemoryPlotData(const tracy::Worker& worker);

// Yields CPU of current thread for a short while, 100 milliseconds.
void YieldCpu();

// Initializes log and parses command line flags.
// Returns all the remaining positional command line arguments.
std::vector<char*> InitializeLogAndParseCommandLine(int argc, char* argv[]);

}  // namespace iree_prof

#endif  // IREE_PROF_OUTPUT_UTILS_H_
