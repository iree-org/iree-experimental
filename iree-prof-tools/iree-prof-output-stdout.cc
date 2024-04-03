// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-prof-tools/iree-prof-output-stdout.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "iree-prof-tools/iree-prof-output-utils.h"
#include "third_party/abseil-cpp/absl/container/flat_hash_map.h"
#include "third_party/abseil-cpp/absl/log/check.h"
#include "third_party/abseil-cpp/absl/status/status.h"
#include "third_party/abseil-cpp/absl/strings/ascii.h"
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/abseil-cpp/absl/strings/string_view.h"
#include "third_party/tracy/public/common/TracyProtocol.hpp"
#include "third_party/tracy/server/TracyWorker.hpp"

namespace iree_prof {

class IreeProfOutputStdout::OutputStream {
 public:
  virtual ~OutputStream() = default;

  virtual OutputStream& operator<<(size_t value) = 0;
  virtual OutputStream& operator<<(absl::string_view value) = 0;
};

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
template <typename T, typename U = typename ZoneTypeHelper<T>::EventType>
absl::flat_hash_map<int, int64_t> GetThreadDurations(
    const tracy::Worker& worker,
    const tracy::unordered_flat_map<int16_t, T>& zones,
    const std::regex& thread_regex) {
  absl::flat_hash_map<int, int64_t> thread_durations;
  for (const auto& z : zones) {
    for (const auto& t : z.second.zones) {
      auto tid = GetThreadId(t);
      if (!thread_durations.contains(tid)) {
        thread_durations[tid] = GetThreadDuration<U>(worker, tid);
      }
    }
  }

  // Filters threads matched with thread_regex.
  absl::flat_hash_map<int, int64_t> filtered_thread_durations;
  for (const auto& d : thread_durations) {
    if (std::regex_search(GetThreadName<U>(worker, d.first), thread_regex)) {
      filtered_thread_durations[d.first] = d.second;
    }
  }

  return filtered_thread_durations;
}

bool HasSubstr(const absl::string_view str,
               const std::vector<std::string>& substrs) {
  for (const auto& s : substrs) {
    if (str.find(s) != str.npos) {
      return true;
    }
  }
  return false;
}

template <typename T>
struct Stat {
  absl::string_view name;

  // Total count of objects running on threads filtered.
  int64_t total_count;

  // Total duration of objects running on threads filtered.
  int64_t total_duration;

  // Duration per thread filtered. Sum of durations must be the same with
  // total_duration.
  absl::flat_hash_map<int, int64_t> duration_per_thread;
};

template <typename T>
void SortStats(std::vector<Stat<T>>& stats) {
  std::sort(stats.begin(), stats.end(),
            [](const Stat<T>& a, const Stat<T>& b) {
              // Sort in a descending order.
              return a.total_duration > b.total_duration;
            });
}

// Returns zones running on threads filtered, and sorted by the total duration.
template <typename T>
std::vector<Stat<T>> GetZoneStatsFilteredAndSorted(
    const tracy::Worker& worker,
    const tracy::unordered_flat_map<int16_t, T>& zones,
    const std::vector<std::string>& zone_substrs,
    const std::optional<std::regex>& zone_regex,
    const absl::flat_hash_map<int, int64_t>& thread_durations) {
  std::vector<Stat<T>> zone_stats_filtered;
  absl::flat_hash_map<absl::string_view, int> zone_stats_filtered_index;
  for (const auto& z : zones) {
    for (const auto& t : z.second.zones) {
      const char* zone_name = worker.GetZoneName(*t.Zone());
      bool matched =
          !zone_substrs.empty() && HasSubstr(zone_name, zone_substrs) ||
          zone_regex && std::regex_search(zone_name, *zone_regex);
      if (!matched) {
        continue;
      }

      if (!zone_stats_filtered_index.contains(zone_name)) {
        zone_stats_filtered_index[zone_name] = zone_stats_filtered.size();
        zone_stats_filtered.emplace_back(Stat<T>{.name = zone_name});
      }

      auto& stat = zone_stats_filtered[zone_stats_filtered_index[zone_name]];
      auto tid = GetThreadId(t);
      if (thread_durations.contains(tid)) {
        ++stat.total_count;
        auto duration = GetEventDuration(*t.Zone());
        stat.total_duration += duration;
        stat.duration_per_thread[tid] += duration;
      }
    }
  }

  SortStats(zone_stats_filtered);
  return zone_stats_filtered;
};

// Gets ML operation name from zone name. Zone names of ML operations follow the
// format of "<prefix>_dispatch_<depth>_<op>_<dimension>_<type>".
// Returns an empty string if it is not an ML operation.
absl::string_view GetOpName(absl::string_view zone_name) {
  // Append "?" after ".+" in op match for non-greedy repetition.
  static const std::regex op_regex(
      "^.*_dispatch_[0-9]+_(.+?)(_[0-9x]+)?(_[if][0-9]+)?$");

  std::cmatch result;
  if (std::regex_match(zone_name.begin(), zone_name.end(), result, op_regex)) {
    return absl::string_view(result[1].first, result[1].length());
  }
  return "";
}

template <typename T>
std::vector<Stat<T>> GetPerOpStats(const std::vector<Stat<T>>& zone_stats) {
  std::vector<Stat<T>> op_stats;
  absl::flat_hash_map<absl::string_view, int> op_stats_index;
  for (const auto& z : zone_stats) {
    auto op_name = GetOpName(z.name);
    if (op_name.empty()) {
      continue;
    }

    if (!op_stats_index.contains(op_name)) {
      op_stats_index[op_name] = op_stats.size();
      op_stats.emplace_back(Stat<T>{.name = op_name});
    }

    auto& stat = op_stats[op_stats_index[op_name]];
    stat.total_count += z.total_count;
    stat.total_duration += z.total_duration;
    for (const auto& d : z.duration_per_thread) {
      stat.duration_per_thread[d.first] += d.second;
    }
  }

  SortStats(op_stats);
  return op_stats;
}

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

// Fills the output table with stat information.
template <typename T, typename U = typename ZoneTypeHelper<T>::EventType>
void FillOutputTableRowWithStat(
    const tracy::Worker& worker,
    const Stat<T>& stat,
    int64_t total_duration,
    const absl::flat_hash_map<int, int64_t>& thread_durations,
    IreeProfOutputStdout::DurationUnit unit,
    const std::vector<std::string>& headers,
    std::vector<std::string>& output_row) {
  output_row[0] = stat.name;
  output_row[1] = absl::StrCat(stat.total_count);
  output_row[2] = absl::StrCat(
      GetDurationStr(stat.total_duration, unit),
      GetPercentage(stat.total_duration, total_duration));
  for (auto it : stat.duration_per_thread) {
    output_row[GetColOfThread(headers, GetThreadName<U>(worker, it.first))] =
        absl::StrCat(GetDurationStr(it.second, unit),
                     GetPercentage(it.second, thread_durations.at(it.first)));
  }
}

// Builds the output table.
// 1st row is for headers, 2nd row is for durations per thread.
// 1st col is for names, 2nd is for counts, 3rd is for total durations.
template <typename T, typename U = typename ZoneTypeHelper<T>::EventType>
std::vector<std::vector<std::string>> BuildOutputTable(
    const tracy::Worker& worker,
    const std::vector<Stat<T>>& stats,
    int64_t total_duration,
    const absl::flat_hash_map<int, int64_t>& thread_durations,
    IreeProfOutputStdout::DurationUnit unit,
    absl::string_view stat_type) {
  auto num_rows = stats.size() + 2;
  auto num_cols = thread_durations.size() + 3;

  std::vector<std::vector<std::string>> output_table(num_rows);

  auto& headers = output_table[0];
  headers.reserve(num_cols);
  headers.push_back(std::string(stat_type));
  headers.push_back("Count");
  headers.push_back("Total");
  for (const auto& it : thread_durations) {
    headers.push_back(GetThreadName<U>(worker, it.first));
  }
  std::sort(headers.begin() + 3, headers.end());

  auto& totals = output_table[1];
  totals.resize(num_cols);
  totals[0] = "Duration";
  // totals[1] is empty since count is not a duration.
  totals[2] = absl::StrCat(GetDurationStr(total_duration, unit), "(100%)");
  for (const auto& it : thread_durations) {
    totals[GetColOfThread(headers, GetThreadName<U>(worker, it.first))] =
        absl::StrCat(GetDurationStr(it.second, unit), "(100%)");
  }

  auto output_row = output_table.begin() + 2;
  for (const auto& s : stats) {
    output_row->resize(num_cols);
    FillOutputTableRowWithStat(worker, s, total_duration, thread_durations,
                               unit, headers, *(output_row++));
  }

  return output_table;
}

void OutputTable(const std::vector<std::vector<std::string>>& output_table,
                 absl::string_view header,
                 IreeProfOutputStdout::OutputStream& os) {
  std::vector<int> widths(output_table[0].size());
  for (const auto& row : output_table) {
    for (int i = 0; i < row.size(); ++ i) {
      if (row[i].size() > widths[i]) {
        widths[i] = row[i].size();
      }
    }
  }

  for (const auto& row : output_table) {
    os << header << "      ";
    for (int i = 0; i < row.size(); ++ i) {
      os << row[i] << std::string(widths[i] - row[i].size() + 1, ' ');
    }
    os << "\n";
  }
}

// Output tabulated information of tracy zones filtered with |zone_substrs|,
// |zone_regex| and |thread_regex|.
template <typename T>
void OutputToStream(const tracy::Worker& worker,
                    const tracy::unordered_flat_map<int16_t, T>& zones,
                    bool output_zone_stats,
                    bool output_per_op_stats,
                    const std::vector<std::string>& zone_substrs,
                    const std::optional<std::regex>& zone_regex,
                    const std::regex& thread_regex,
                    absl::string_view header,
                    IreeProfOutputStdout::DurationUnit unit,
                    IreeProfOutputStdout::OutputStream& os) {
  if (zones.empty()) {
    return;
  }

  auto thread_durations = GetThreadDurations(worker, zones, thread_regex);
  if (thread_durations.empty()) {
    return;
  }

  int64_t total = 0;
  for (const auto& it : thread_durations) {
    total += it.second;
  }

  auto zone_stats_filtered = GetZoneStatsFilteredAndSorted(
      worker, zones, zone_substrs, zone_regex, thread_durations);
  if (output_zone_stats) {
    os << header << "   Zone Stats" << ": "
       << zone_stats_filtered.size() << "\n";
    auto zone_output_table =
        BuildOutputTable(worker, zone_stats_filtered, total, thread_durations,
                         unit, "Zone");
    OutputTable(zone_output_table, header, os);
  }

  if (output_per_op_stats) {
    auto per_op_stats = GetPerOpStats(zone_stats_filtered);
    if (!per_op_stats.empty()) {
      os << header << " Per-OP Stats" << ": " << per_op_stats.size() << "\n";
      auto per_op_output_table = BuildOutputTable(worker, per_op_stats, total,
                                                  thread_durations, unit, "OP");
      OutputTable(per_op_output_table, header, os);
    }
  }
}

class StdOutStream : public IreeProfOutputStdout::OutputStream {
 public:
  StdOutStream() = default;
  ~StdOutStream() override = default;

  OutputStream& operator<<(size_t value) override {
    std::cout << value;
    return *this;
  }

  OutputStream& operator<<(absl::string_view value) override {
    std::cout << value;
    return *this;
  }
};

class CombinedStream : public IreeProfOutputStdout::OutputStream {
 public:
  CombinedStream(std::unique_ptr<OutputStream> a,
                 std::unique_ptr<OutputStream> b)
      : a_(std::move(a)), b_(std::move(b)) {}

  ~CombinedStream() override = default;

  OutputStream& operator<<(size_t value) override {
    *a_ << value;
    *b_ << value;
    return *this;
  }

  OutputStream& operator<<(absl::string_view value) override {
    *a_ << value;
    *b_ << value;
    return *this;
  }

 private:
  std::unique_ptr<OutputStream> a_;
  std::unique_ptr<OutputStream> b_;
};

// Output values into a Comma-Separated-Values file. Note that it relies on
// the client output delimiters, i.e. ':', ',' and '=' separately from values
// for easier processing.
class CsvStream : public IreeProfOutputStdout::OutputStream {
 public:
  CsvStream(absl::string_view csv_file_path,
            IreeProfOutputStdout::DurationUnit unit)
      : fout_(std::ofstream(std::string(csv_file_path).c_str())),
        unit_str_(GetUnitStr(unit)),
        is_table_header_(false),
        column_idx_(0) {}

  ~CsvStream() override { fout_.close(); }

  OutputStream& operator<<(size_t value) override {
    fout_ << value;
    return *this;
  }

  OutputStream& operator<<(absl::string_view value) override {
    if (value.empty()) {
      // If value is empty, it's an empty column value. If the column is for
      // duration stat, add one more comma since duration stat needs 2 columns.
      ++column_idx_;
      if (column_idx_ > 3) {
        fout_ << ", ";
        ++column_idx_;  // Increase column_idx as if it emits an empty string.
      }
      return *this;
    }

    // |value| may be
    // 1) '\n',
    // 2) a explicit delimiter, i.e. ':', '=', or ',',
    // 3) a column delimiter, i.e. non-empty whitespaces, or
    // 4) just normal value to output.

    if (value == "\n") {
      fout_ << "\n";
      is_table_header_ = false;
      column_idx_ = 0;
      return *this;
    }

    auto value_no_space = absl::StripAsciiWhitespace(value);
    if (IsDelimiter(value_no_space)) {
      fout_ << ", ";
      return *this;
    }
    CHECK(!value_no_space.empty());

    if (IsHeader(value_no_space)) {
      return *this;
    }

    // Output duration stats into 2 columns, one in units, one in percentage.
    auto paren_pos = value_no_space.find('(');
    if (paren_pos != value_no_space.npos) {
      CHECK_EQ(
          value_no_space.substr(paren_pos - unit_str_.size(), unit_str_.size()),
          unit_str_);
      CHECK_EQ(*value_no_space.rbegin(), ')');
      fout_ << value_no_space.substr(0, paren_pos - unit_str_.size()) << ", "
            << value_no_space.substr(paren_pos + 1,
                                     value_no_space.size() - paren_pos - 2);
      column_idx_ += 2;
      return *this;
    }

    fout_ << value_no_space;
    ++column_idx_;

    if (value_no_space == "Total") {
      is_table_header_ = true;
    }

    // For table headers, print with unit_str and another command for the column
    // of percentage.
    if (is_table_header_) {
      fout_ << "(" << unit_str_ << "), ";
      ++column_idx_;
    }

    return *this;
  }

 private:
  static std::string GetUnitStr(IreeProfOutputStdout::DurationUnit unit) {
    // Remove '0'.
    return GetDurationStr(0, unit).substr(1);
  }

  bool IsDelimiter(absl::string_view str) {
    // If empty, i.e. all whitespaces before being stripped, it's a column
    // delimiter.
    return str.empty() || str == ":" || str == "," || str == "=";
  }

  bool IsHeader(absl::string_view str) {
    return *str.begin() == '[' && *str.rbegin() == ']';
  }

  std::ofstream fout_;
  const std::string unit_str_;
  bool is_table_header_;
  int column_idx_;
};

std::unique_ptr<IreeProfOutputStdout::OutputStream> CreateOutputStream(
    bool output_stdout,
    absl::string_view csv_file_path,
    IreeProfOutputStdout::DurationUnit unit) {
  std::unique_ptr<IreeProfOutputStdout::OutputStream> os;
  if (output_stdout) {
    os = std::make_unique<StdOutStream>();
  }

  if (!csv_file_path.empty()) {
    auto csv_os = std::make_unique<CsvStream>(csv_file_path, unit);
    if (os) {
      os = std::make_unique<CombinedStream>(std::move(os), std::move(csv_os));
    } else {
      os = std::move(csv_os);
    }
  }

  return os;
}

}  // namespace

IreeProfOutputStdout::IreeProfOutputStdout(
    bool output_stdout,
    absl::string_view csv_file_path,
    bool output_zone_stats,
    bool output_per_op_stats,
    const std::vector<std::string>& zone_substrs,
    const std::string& zone_regex,
    const std::string& thread_regex,
    DurationUnit unit)
    : output_zone_stats_(output_zone_stats),
      output_per_op_stats_(output_per_op_stats),
      zone_substrs_(zone_substrs),
      zone_regex_(zone_regex.empty() ? std::nullopt
                                     : std::optional<std::regex>(zone_regex)),
      thread_regex_(thread_regex),
      unit_(unit),
      os_(CreateOutputStream(output_stdout, csv_file_path, unit)) {
  CHECK(os_) << "Either output_stdout or csv_file_path should be true or "
             << "not empty.";
}

IreeProfOutputStdout::~IreeProfOutputStdout() = default;

absl::Status IreeProfOutputStdout::Output(tracy::Worker& worker) {
  os() << "[TRACY    ]" << " Capture Name" << ": "
       << worker.GetCaptureName() << "\n";
  os() << "[TRACY    ]" << "     Cpu Arch" << ": "
       << ArchToString(worker.GetCpuArch()) << "\n";

  if (!worker.GetThreadData().empty()) {
    os() << "[TRACY    ]" << "\n";
    os() << "[TRACY-CPU]" << "  CPU Threads" << ": "
         << worker.GetThreadData().size() << "\n";
    os() << "[TRACY-CPU]" << "    CPU Zones" << ": "
         << worker.GetZoneCount() << "\n";
    OutputToStream(worker, worker.GetSourceLocationZones(), output_zone_stats_,
                   output_per_op_stats_, zone_substrs_, zone_regex_,
                   thread_regex_, "[TRACY-CPU]", unit_, os());
  }

  if (!worker.GetGpuData().empty()) {
    os() << "[TRACY    ]" << "\n";
    os() << "[TRACY-GPU]" << " GPU Contexts" << ": "
         << worker.GetGpuData().size();
    for (const auto& d : worker.GetGpuData()) {
      os() << ", " << worker.GetString(d->name);
    }
    os() << "\n";
    os() << "[TRACY-GPU]" << "    GPU Zones" << ": "
         << worker.GetGpuZoneCount() << "\n";
    OutputToStream(worker, worker.GetGpuSourceLocationZones(),
                   output_zone_stats_, output_per_op_stats_,  zone_substrs_,
                   zone_regex_, thread_regex_, "[TRACY-GPU]", unit_, os());
  }

  if (!worker.GetMemNameMap().empty()) {
    os() << "[TRACY    ]" << "\n";
    const auto* p = GetMemoryPlotData(worker);
    if (p) {
      os() << "[TRACY-MEM]" << "   Max Memory" << ": "
           << MemToString(p->max) << "\n";
    }
    os() << "[TRACY-MEM]" << "   Allocators" << ": "
         << worker.GetMemNameMap().size() << "\n";
    for (const auto& m : worker.GetMemNameMap()) {
      os() << "[TRACY-MEM]" << "        Alloc" << ": " << "name" << "="
           << (m.first == 0 ? "default" : worker.GetString(m.first)) << ", "
           << "events" << "=" << m.second->data.size() << "\n";
    }
  }

  return absl::OkStatus();
}

}  // namespace iree_prof
