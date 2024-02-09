// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-prof-tools/iree-prof-output-stdout.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

#include "iree-prof-tools/iree-prof-output-utils.h"
#include "third_party/abseil-cpp/absl/container/flat_hash_map.h"
#include "third_party/abseil-cpp/absl/status/status.h"
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/abseil-cpp/absl/strings/string_view.h"
#include "third_party/tracy/public/common/TracyProtocol.hpp"
#include "third_party/tracy/server/TracyWorker.hpp"

namespace iree_prof {
namespace {

const char* ArchToString(tracy::CpuArchitecture arch) {
  switch (arch) {
    case tracy::CpuArchUnknown: return "Unknown";
    case tracy::CpuArchX86: return "x86";
    case tracy::CpuArchX64: return "x86_64";
    case tracy::CpuArchArm32: return "arm";
    case tracy::CpuArchArm64: return "aarch64";
    default: return "Unknown";
  }
}

std::string MemToString(double mem_usage) {
  if (mem_usage >= 995 * 1000 * 1000) {
    return absl::StrCat(floor(mem_usage / 1000 / 1000 / 10 + 0.5) / 100,
                        " GBytes");
  } else if (mem_usage >= 995 * 1000) {
    return absl::StrCat(floor(mem_usage / 1000 / 10 + 0.5) / 100, " MBytes");
  } else if (mem_usage >= 995) {
    return absl::StrCat(floor(mem_usage / 10 + 0.5) / 100, " KBytes");
  }
  return absl::StrCat(mem_usage, " Bytes");
}

// Whether |substrs| includes a substring of |str|.
bool HasSubstr(absl::string_view str, const std::vector<std::string>& substrs) {
  return std::find_if(
             substrs.begin(), substrs.end(),
             [str](const std::string& s) { return str.find(s) != str.npos; })
             != substrs.end();
}

// Returns a string of duration with unit.
std::string GetDurationStr(int64_t duration_ns,
                           IreeProfOutputStdout::DurationUnit unit) {
  switch (unit) {
    case IreeProfOutputStdout::DurationUnit::kNanoseconds:
      return absl::StrCat(duration_ns, "ns");
    case IreeProfOutputStdout::DurationUnit::kMicroseconds:
      return absl::StrCat(static_cast<double>(duration_ns) / 1000, "us");
    case IreeProfOutputStdout::DurationUnit::kSeconds:
      return absl::StrCat(static_cast<double>(duration_ns) / 1000000000, "s");
    case IreeProfOutputStdout::DurationUnit::kMilliseconds:
    default:
      return absl::StrCat(static_cast<double>(duration_ns) / 1000000, "ms");
  }
}

// Returns the duration per thread, i.e merged durations of all top-level zones
// running on each thread.
template <typename T>
absl::flat_hash_map<int, int64_t> GetThreadDurations(
    const tracy::Worker& worker,
    const tracy::unordered_flat_map<int16_t, T>& zones,
    const std::vector<std::string>& thread_substrs) {
  absl::flat_hash_map<int, int64_t> thread_durations;
  for (const auto& z : zones) {
    for (const auto& t : z.second.zones) {
      auto tid = GetThreadId(t);
      if (!thread_durations.contains(tid)) {
        thread_durations[tid] = GetThreadDuration<T>(worker, tid);
      }
    }
  }

  // Filters threads matched with substrings in thread_substrs if not empty.
  // If empty, add all.
  if (!thread_substrs.empty()) {
    absl::flat_hash_map<int, int64_t> filtered_thread_durations;
    for (const auto& d : thread_durations) {
      if (HasSubstr(GetThreadName<T>(worker, d.first), thread_substrs)) {
        filtered_thread_durations[d.first] = d.second;
      }
    }
    thread_durations.swap(filtered_thread_durations);
  }
  return thread_durations;
}

template <typename T>
struct Zone {
  const char* name;
  const T* zone;

  // Total count of zones running on threads filtered. It must be the same to
  // zone->zones.size() if no threads are filtered out.
  int64_t total_count;

  // Total duration of zones running on threads filtered. It must be the same to
  // zone->total if no threads are filtered out.
  int64_t total_duration;
};

// Returns zones running on threads filtered, and sorted by the total duration.
template <typename T>
std::vector<Zone<T>> GetZonesFilteredAndSorted(
    const tracy::Worker& worker,
    const tracy::unordered_flat_map<int16_t, T>& zones,
    const std::vector<std::string>& zone_substrs,
    const absl::flat_hash_map<int, int64_t>& thread_durations) {
  std::vector<Zone<T>> zones_filtered;
  for (const auto& z : zones) {
    const char* zone_name = GetZoneName(worker, z.first);
    if (!HasSubstr(zone_name, zone_substrs)) {
      continue;
    }

    int64_t total_count = 0;
    int64_t total_duration = 0;
    for (const auto& t : z.second.zones) {
      if (thread_durations.contains(GetThreadId(t))) {
        ++total_count;
        total_duration += GetEventDuration(*t.Zone());
      }
    }

    if (total_count == 0 || total_duration == 0) {
      continue;
    }

    zones_filtered.emplace_back(
        Zone<T>{zone_name, &z.second, total_count, total_duration});
  }

  std::sort(zones_filtered.begin(), zones_filtered.end(),
            [](const Zone<T>& a, const Zone<T>& b) {
              // Sort in a descending order.
              return a.total_duration > b.total_duration;
            });

  return zones_filtered;
};

// Returns the index of |thread_name| in |headers| which is effectivtly the
// column index of the given thread in the output table.
int GetColOfThread(const std::vector<std::string>& headers,
                   absl::string_view thread_name) {
  for (int i = 3; i < headers.size(); ++i) {
    if (headers[i] == thread_name) {
      return i;
    }
  }
  return headers.size();  // Return a wrong index intentially.
}

// Returns the string of percentage of |num| in |total|.
std::string GetPercentage(int64_t num, int64_t total) {
  double percentage = static_cast<double>(num * 10000 / total) / 100;
  return absl::StrCat("(", percentage, "%)");
}

// Fills the output table with zone information.
template <typename T>
void FillOutputTableRowWithZone(
    const tracy::Worker& worker,
    const Zone<T>& zone,
    int64_t total_duration,
    const absl::flat_hash_map<int, int64_t>& thread_durations,
    IreeProfOutputStdout::DurationUnit unit,
    const std::vector<std::string>& headers,
    std::vector<std::string>& output_row) {
  absl::flat_hash_map<int, int64_t> ns_per_thread;
  for (const auto& t : zone.zone->zones) {
    auto tid = GetThreadId(t);
    if (thread_durations.contains(tid)) {
      ns_per_thread[tid] += GetEventDuration(*t.Zone());
    }
  }

  output_row[0] = zone.name;
  output_row[1] = absl::StrCat(zone.total_count);
  output_row[2] = absl::StrCat(
      GetDurationStr(zone.total_duration, unit),
      GetPercentage(zone.total_duration, total_duration));
  for (auto it : ns_per_thread) {
    output_row[GetColOfThread(headers, GetThreadName<T>(worker, it.first))] =
        absl::StrCat(GetDurationStr(it.second, unit),
                     GetPercentage(it.second, thread_durations.at(it.first)));
  }
}

// Builds the output table.
// 1st row is for headers, 2nd row is for durations of zones per thread.
// 1st col is for zone names, 2nd is for counts, 3rd is for total durations.
template <typename T>
std::vector<std::vector<std::string>> BuildOutputTable(
    const tracy::Worker& worker,
    const std::vector<Zone<T>>& zones,
    int64_t total_duration,
    const absl::flat_hash_map<int, int64_t>& thread_durations,
    IreeProfOutputStdout::DurationUnit unit) {
  auto num_rows = zones.size() + 2;
  auto num_cols = thread_durations.size() + 3;

  std::vector<std::vector<std::string>> output_table(num_rows);

  auto& headers = output_table[0];
  headers.reserve(num_cols);
  headers.push_back("Zone");
  headers.push_back("Count");
  headers.push_back("Total");
  for (const auto& it : thread_durations) {
    headers.push_back(GetThreadName<T>(worker, it.first));
  }
  std::sort(headers.begin() + 3, headers.end());

  auto& totals = output_table[1];
  totals.resize(num_cols);
  totals[0] = "Duration";
  // totals[1] is empty since count is not a duration.
  totals[2] = GetDurationStr(total_duration, unit);
  for (const auto& it : thread_durations) {
    totals[GetColOfThread(headers, GetThreadName<T>(worker, it.first))] =
        GetDurationStr(it.second, unit);
  }

  auto output_row = output_table.begin() + 2;
  for (const auto& z : zones) {
    output_row->resize(num_cols);
    FillOutputTableRowWithZone(worker, z, total_duration, thread_durations,
                               unit, headers, *(output_row++));
  }

  return output_table;
}

// Output tabulated information of tracy zones filtered with |zone_substrs| and
// |thread_substrs|.
template <typename T>
void OutputToStdout(
    const tracy::Worker& worker,
    const tracy::unordered_flat_map<int16_t, T>& zones,
    const std::vector<std::string>& zone_substrs,
    const std::vector<std::string>& thread_substrs,
    absl::string_view header,
    IreeProfOutputStdout::DurationUnit unit) {
  if (zones.empty()) {
    return;
  }

  auto thread_durations = GetThreadDurations(worker, zones, thread_substrs);
  if (thread_durations.empty()) {
    return;
  }

  int64_t total = 0;
  for (const auto& it : thread_durations) {
    total += it.second;
  }

  auto zones_filtered = GetZonesFilteredAndSorted(worker, zones, zone_substrs,
                                                  thread_durations);
  auto output_table = BuildOutputTable(worker, zones_filtered,
                                       total, thread_durations, unit);

  std::vector<int> widths(output_table[0].size());
  for (const auto& row : output_table) {
    for (int i = 0; i < row.size(); ++ i) {
      if (row[i].size() > widths[i]) {
        widths[i] = row[i].size();
      }
    }
  }

  for (const auto& row : output_table) {
    std::cout << header << "      ";
    for (int i = 0; i < row.size(); ++ i) {
      std::cout << row[i] << std::string(widths[i] - row[i].size() + 1, ' ');
    }
    std::cout << "\n";
  }
}

}  // namespace

IreeProfOutputStdout::IreeProfOutputStdout(
    const std::vector<std::string>& zone_substrs,
    const std::vector<std::string>& thread_substrs,
    DurationUnit unit)
    : zone_substrs_(zone_substrs),
      thread_substrs_(thread_substrs),
      unit_(unit) {}

IreeProfOutputStdout::~IreeProfOutputStdout() = default;

absl::Status IreeProfOutputStdout::Output(tracy::Worker& worker) {
  std::cout << "[TRACY    ] Capture Name: " << worker.GetCaptureName() << "\n";
  std::cout << "[TRACY    ]     Cpu Arch: " << ArchToString(worker.GetCpuArch())
            << "\n";

  if (!worker.GetThreadData().empty()) {
    std::cout << "[TRACY    ]\n";
    std::cout << "[TRACY-CPU]  CPU Threads: " << worker.GetThreadData().size()
              << "\n";
    std::cout << "[TRACY-CPU]    CPU Zones: " << worker.GetZoneCount() << "\n";
    OutputToStdout(worker, worker.GetSourceLocationZones(), zone_substrs_,
                   thread_substrs_, "[TRACY-CPU]", unit_);
  }

  if (!worker.GetGpuData().empty()) {
    std::cout << "[TRACY    ]\n";
    std::cout << "[TRACY-GPU] GPU Contexts: " << worker.GetGpuData().size();
    for (const auto& d : worker.GetGpuData()) {
      std::cout << ", " << worker.GetString(d->name);
    }
    std::cout << "\n";
    std::cout << "[TRACY-GPU]    GPU Zones: " << worker.GetGpuZoneCount()
              << "\n";
    OutputToStdout(worker, worker.GetGpuSourceLocationZones(), zone_substrs_,
                   thread_substrs_, "[TRACY-GPU]", unit_);
  }

  if (!worker.GetMemNameMap().empty()) {
    std::cout << "[TRACY    ]\n";
    const auto* p = GetMemoryPlotData(worker);
    if (p) {
      std::cout << "[TRACY-MEM]   Max Memory: " << MemToString(p->max) << "\n";
    }
    std::cout << "[TRACY-MEM]   Allocators: " << worker.GetMemNameMap().size()
              << "\n";
    for (const auto& m : worker.GetMemNameMap()) {
      std::cout << "[TRACY-MEM]        Alloc: name="
                << (m.first == 0 ? "default" : worker.GetString(m.first))
                << ", events=" << m.second->data.size()
                << "\n";
    }
  }

  return absl::OkStatus();
}

}  // namespace iree_prof
