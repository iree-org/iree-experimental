// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-prof-tools/iree-prof-output-chrome.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "iree-prof-tools/iree-prof-output-utils.h"
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/abseil-cpp/absl/strings/str_join.h"
#include "third_party/abseil-cpp/absl/strings/string_view.h"
#include "third_party/tracy/server/TracyWorker.hpp"

namespace iree_prof {
namespace {

// Chrome tracing viewer (https://github.com/catapult-project/catapult) format
// is described in
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit?usp=sharing.

constexpr int kPidFake = 0;

constexpr absl::string_view kTypeMetadata = "M";
constexpr absl::string_view kTypeEventStart = "B";
constexpr absl::string_view kTypeEventEnd = "E";

std::string GetSourceFileLineOrUnknown(const tracy::Worker& worker,
                                       int16_t source_location_id) {
  auto file_line = GetSourceFileLine(worker, source_location_id);
  return file_line.empty() ? "unknown" : file_line;
}

// Returns 2 ordered vector of memory events according to alloc/free timestamp.
void SortMemEvents(const tracy::Worker& worker,
                   std::vector<const tracy::MemEvent*>& sorted_by_alloc,
                   std::vector<const tracy::MemEvent*>& sorted_by_free) {
  for (const auto& m : worker.GetMemNameMap()) {
    for (const auto& e : m.second->data) {
      sorted_by_alloc.push_back(&e);
      sorted_by_free.push_back(&e);
    }
  }

  std::sort(sorted_by_alloc.begin(), sorted_by_alloc.end(),
            [](const tracy::MemEvent* a, const tracy::MemEvent* b) {
              return a->TimeAlloc() < b->TimeAlloc();
            });
  std::sort(sorted_by_free.begin(), sorted_by_free.end(),
            [](const tracy::MemEvent* a, const tracy::MemEvent* b) {
              return a->TimeFree() < b->TimeFree();
            });
}

// Gets mem stats from a zone event.
struct ZoneMemStat {
  int64_t num_allocs;   // # of memory allocations within the zone.
  int64_t num_frees;    // # of memory deallocations within the zone.
  int64_t size_allocs;  // Sum of memory sizes of allocations.
  int64_t size_frees;   // Sum of memory sizes of deallocations.
};

ZoneMemStat GetZoneMemStat(
    const std::vector<const tracy::MemEvent*>& sorted_by_alloc,
    const std::vector<const tracy::MemEvent*>& sorted_by_free,
    uint16_t thread_id,
    int64_t start,
    int64_t end) {
  ZoneMemStat stat = {};

  auto it_alloc =
      std::lower_bound(sorted_by_alloc.begin(), sorted_by_alloc.end(), start,
                       [](const tracy::MemEvent* a, int64_t start) {
                         return a->TimeAlloc() < start;
                       });
  for (; it_alloc != sorted_by_alloc.end(); ++it_alloc) {
    if (end < (*it_alloc)->TimeAlloc()) {
      break;
    }
    if (thread_id == (*it_alloc)->ThreadAlloc()) {
      ++stat.num_allocs;
      stat.size_allocs += (*it_alloc)->Size();
    }
  }

  auto it_free =
      std::lower_bound(sorted_by_free.begin(), sorted_by_free.end(), start,
                       [](const tracy::MemEvent* a, int64_t start) {
                         return a->TimeFree() < start;
                       });
  for (; it_free != sorted_by_free.end(); ++it_free) {
    if (end < (*it_free)->TimeFree()) {
      break;
    }
    if (thread_id == (*it_free)->ThreadFree()) {
      ++stat.num_frees;
      stat.size_frees += (*it_free)->Size();
    }
  }

  return stat;
}

// Outputs a zone event as a JSON array entry.
void OutputEvent(absl::string_view name,
                 std::vector<std::string> categories,
                 absl::string_view event_type,
                 int64_t timestamp_ns,
                 int thread_id,
                 std::vector<std::string> args,
                 std::ofstream& fout) {
  fout << "{";
  if (!name.empty()) {
    fout << "\"name\": \"" << name << "\", ";
  }
  if (!categories.empty()) {
    fout << "\"cat\": \"" << absl::StrJoin(categories, "\", \"") << "\", ";
  }
  fout << "\"ph\": \"" << event_type << "\", ";
  fout << "\"ts\": " << static_cast<double>(timestamp_ns) / 1000 << ", ";
  fout << "\"pid\": " << kPidFake << ", ";
  fout << "\"tid\": " << thread_id;
  if (!args.empty()) {
    fout << ", \"args\": {" << absl::StrJoin(args, ", ") << "}";
  }
  fout << "}";
}

// Returns a JSON key-value pair used for an event argument.
std::string ToArgField(absl::string_view key, absl::string_view str_value) {
  return absl::StrCat("\"", key, "\": \"", str_value, "\"");
}

std::string ToArgField(absl::string_view key, int64_t int_value) {
  return absl::StrCat("\"", key, "\": ", int_value);
}

// Forward decl.
template <typename T>
void OutputTimeline(const tracy::Worker& worker,
                    const std::vector<const tracy::MemEvent*>& sorted_by_alloc,
                    const std::vector<const tracy::MemEvent*>& sorted_by_free,
                    uint16_t thread_id,
                    const tracy::Vector<tracy::short_ptr<T>>& timeline,
                    std::ofstream& fout);

// Outputs the zone events from a timeline which might be interleaved by zone
// events of the child timelines.
template <typename T>
void RealOutputTimeline(
    const tracy::Worker& worker,
    const std::vector<const tracy::MemEvent*>& sorted_by_alloc,
    const std::vector<const tracy::MemEvent*>& sorted_by_free,
    uint16_t thread_id,
    const tracy::Vector<T>& timeline,
    std::ofstream& fout) {
  for (const auto& e : timeline) {
    const auto& zone_event = GetEvent(e);
    auto zone_id = zone_event.SrcLoc();
    auto start = GetEventStart(zone_event);
    auto end = GetEventEnd(zone_event);

    std::vector<std::string> args;
    args.push_back(ToArgField("source", GetSourceFileLine(worker, zone_id)));

    auto mem_stat =
        GetZoneMemStat(sorted_by_alloc, sorted_by_free, thread_id, start, end);
    args.push_back(ToArgField("num_allocs", mem_stat.num_allocs));
    args.push_back(ToArgField("num_frees", mem_stat.num_frees));
    args.push_back(ToArgField("size_allocs", mem_stat.size_allocs));
    args.push_back(ToArgField("size_frees", mem_stat.size_frees));

    fout << ",\n";
    OutputEvent(worker.GetZoneName(zone_event), {}, kTypeEventStart, start,
                thread_id, args, fout);

    auto* children = GetEventChildren(worker, zone_event);
    if (children) {
      OutputTimeline(worker, sorted_by_alloc, sorted_by_free, thread_id,
                     *children, fout);
    }

    fout << ",\n";
    OutputEvent("", {}, kTypeEventEnd, end, thread_id, {}, fout);
  }
}

// Outputs the zone events from a timeline and its child timelines by the help
// of RealOutputTimeline(). It's to differentiate templates of tracy::Vector<T>
// and ones of tracy::Vector<tracy::short_ptr<T>>.
template <typename T>
void OutputTimeline(const tracy::Worker& worker,
                    const std::vector<const tracy::MemEvent*>& sorted_by_alloc,
                    const std::vector<const tracy::MemEvent*>& sorted_by_free,
                    uint16_t thread_id,
                    const tracy::Vector<tracy::short_ptr<T>>& timeline,
                    std::ofstream& fout) {
  if (timeline.is_magic()) {
    RealOutputTimeline(worker, sorted_by_alloc, sorted_by_free, thread_id,
                       *reinterpret_cast<const tracy::Vector<T>*>(&timeline),
                       fout);
  } else {
    RealOutputTimeline(worker, sorted_by_alloc, sorted_by_free, thread_id,
                       timeline, fout);
  }
}

// Outputs the zone events running on a (CPU or GPU) thread into a chrome
// tracing viewer JSON file.
// A thread is represented by a root timeline and its compressed thread ID.
template <typename T>
void OutputThread(const tracy::Worker& worker,
                  const std::vector<const tracy::MemEvent*>& sorted_by_alloc,
                  const std::vector<const tracy::MemEvent*>& sorted_by_free,
                  uint16_t thread_id,
                  const tracy::Vector<tracy::short_ptr<T>>& timeline,
                  std::ofstream& fout) {
  if (timeline.empty()) {
    return;
  }

  fout << ",\n";
  OutputEvent("thread_name", {}, kTypeMetadata, 0, thread_id,
              {ToArgField("name", GetThreadName<T>(worker, thread_id))}, fout);

  OutputTimeline(worker, sorted_by_alloc, sorted_by_free, thread_id, timeline,
                 fout);
}

// Outputs a tracy worker into a chrome tracing viewer JSON file.
void OutputJson(const tracy::Worker& worker, std::ofstream& fout) {
  fout << "[\n";
  OutputEvent("process_name", {}, kTypeMetadata, 0, 0,
              {ToArgField("name", worker.GetCaptureName())}, fout);

  std::vector<const tracy::MemEvent*> sorted_by_alloc;
  std::vector<const tracy::MemEvent*> sorted_by_free;
  SortMemEvents(worker, sorted_by_alloc, sorted_by_free);

  for (const auto* d : worker.GetThreadData()) {
    OutputThread(worker, sorted_by_alloc, sorted_by_free,
                 const_cast<tracy::Worker*>(&worker)->CompressThread(d->id),
                 d->timeline, fout);
  }

  for (const auto& g : worker.GetGpuData()) {
    for (const auto& d : g->threadData) {
      OutputThread(worker, sorted_by_alloc, sorted_by_free, d.first,
                   d.second.timeline, fout);
    }
  }
  fout << "\n]\n";
}

}  // namespace

IreeProfOutputChrome::IreeProfOutputChrome(absl::string_view output_file_path)
    : output_file_path_(output_file_path) {}

IreeProfOutputChrome::~IreeProfOutputChrome() = default;

absl::Status IreeProfOutputChrome::Output(tracy::Worker& worker) {
  std::ofstream fout(output_file_path_.c_str());
  OutputJson(worker, fout);
  return absl::OkStatus();
}

}  // namespace iree_prof
