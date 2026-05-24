#pragma once

#include <atomic>
#include <optional>
#include <string>

#include "pointcloud/PointAttributes.h"
#include "util/Definitions.h"

struct ProgressReporter;

struct ConverterArguments
{
  std::string source_folder;
  std::string output_folder;
  OutputFormat output_format;
  PointAttributes output_attributes;
  std::optional<std::string> source_projection;
  std::optional<uint32_t> max_depth;
  bool delete_source_files;
  std::atomic<bool>* stop_source = nullptr;
  ProgressReporter* external_progress_reporter = nullptr;
};

/**
 * Run conversion process
 */
void
run_conversion(const ConverterArguments& args);