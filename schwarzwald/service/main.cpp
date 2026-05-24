#include <algorithm>
#include <atomic>
#include <chrono>
#include <experimental/filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "debug/Journal.h"
#include "debug/ProgressReporter.h"
#include "expected.hpp"
#include "io/TileSetWriter.h"
#include "math/AABB.h"
#include "math/Vector3.h"
#include "pointcloud/PointAttributes.h"
#include "pointcloud/Tileset.h"
#include "process/ConverterProcess.h"
#include "process/TilerProcess.h"
#include "terminal/TerminalUI.h"
#include "util/Config.h"
#include "util/Definitions.h"
#include "util/Error.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

// httplib.h includes networking headers that define macros conflicting
// with our enum values (e.g. ADD, DELETE). Undef them so our code compiles.
#include <httplib.h>
#ifdef ADD
#undef ADD
#endif
#ifdef DELETE
#undef DELETE
#endif

namespace rj = rapidjson;

// ---------------------------------------------------------------------------
// Output capture: redirects std::cout / std::cerr into a stringstream
// ---------------------------------------------------------------------------

struct CaptureBuffer : std::streambuf
{
  explicit CaptureBuffer(std::streambuf* original)
    : _original(original)
  {}

  int_type overflow(int_type ch) override
  {
    if (ch != traits_type::eof()) {
      _captured.put(static_cast<char>(ch));
      if (_original)
        _original->sputc(static_cast<char>(ch));
    }
    return ch;
  }

  std::streamsize xsputn(const char* s, std::streamsize n) override
  {
    _captured.write(s, n);
    if (_original)
      _original->sputn(s, n);
    return n;
  }

  std::string str() const { return _captured.str(); }
  void clear() { _captured.str(""); _captured.clear(); }

private:
  std::streambuf* _original;
  std::ostringstream _captured;
};

struct OutputCapture
{
  OutputCapture()
  {
    _cout_buf = std::make_unique<CaptureBuffer>(std::cout.rdbuf());
    _cerr_buf = std::make_unique<CaptureBuffer>(std::cerr.rdbuf());
    _old_cout = std::cout.rdbuf(_cout_buf.get());
    _old_cerr = std::cerr.rdbuf(_cerr_buf.get());
  }

  ~OutputCapture()
  {
    std::cout.rdbuf(_old_cout);
    std::cerr.rdbuf(_old_cerr);
  }

  std::string captured() const
  {
    return _cout_buf->str() + _cerr_buf->str();
  }

  std::unique_ptr<CaptureBuffer> _cout_buf;
  std::unique_ptr<CaptureBuffer> _cerr_buf;
  std::streambuf* _old_cout;
  std::streambuf* _old_cerr;
};

// ---------------------------------------------------------------------------
// Job representation
// ---------------------------------------------------------------------------

struct Job
{
  std::string id;
  std::string type; // "tile" or "convert"
  std::string status; // "pending", "running", "completed", "failed", "stopped"

  std::atomic<bool> stop_requested{ false };
  std::atomic<bool> running{ false };

  std::thread worker;

  std::unique_ptr<UIState> ui_state;
  std::unique_ptr<OutputCapture> output;

  std::string error_message;
  std::string result_summary;

  std::chrono::system_clock::time_point created_at;
  std::chrono::system_clock::time_point started_at;
  std::chrono::system_clock::time_point finished_at;
};

// ---------------------------------------------------------------------------
// Thread-safe job manager
// ---------------------------------------------------------------------------

class JobManager
{
public:
  JobManager() : _next_id(0) {}

  std::string create_job(const std::string& type,
                         std::function<void(Job&)> work_fn)
  {
    auto job = std::make_shared<Job>();
    job->id = generate_id();
    job->type = type;
    job->status = "pending";
    job->created_at = std::chrono::system_clock::now();
    job->ui_state = std::make_unique<UIState>();

    {
      std::lock_guard<std::mutex> lock(_mutex);
      _jobs[job->id] = job;
    }

    job->worker = std::thread([job, work_fn]() {
      job->status = "running";
      job->running = true;
      job->started_at = std::chrono::system_clock::now();

      job->output = std::make_unique<OutputCapture>();

      try {
        work_fn(*job);
        if (job->stop_requested) {
          job->status = "stopped";
        } else {
          job->status = "completed";
        }
      } catch (const std::exception& ex) {
        job->status = "failed";
        job->error_message = ex.what();
      } catch (...) {
        job->status = "failed";
        job->error_message = "Unknown error";
      }

      job->running = false;
      job->finished_at = std::chrono::system_clock::now();
    });

    return job->id;
  }

  std::shared_ptr<Job> get_job(const std::string& id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _jobs.find(id);
    if (it == _jobs.end())
      return nullptr;
    return it->second;
  }

  std::vector<std::shared_ptr<Job>> list_jobs()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<std::shared_ptr<Job>> jobs;
    jobs.reserve(_jobs.size());
    for (auto& kv : _jobs)
      jobs.push_back(kv.second);
    return jobs;
  }

  bool stop_job(const std::string& id)
  {
    auto job = get_job(id);
    if (!job)
      return false;
    if (!job->running)
      return false;
    job->stop_requested = true;
    // Also set stop_source on the progress reporter so the tiler/converter
    // will throw on next progress update
    job->ui_state->get_progress_reporter().stop_source = &job->stop_requested;
    return true;
  }

  bool remove_job(const std::string& id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _jobs.find(id);
    if (it == _jobs.end())
      return false;
    auto& job = it->second;
    if (job->running)
      return false; // can't remove a running job
    if (job->worker.joinable())
      job->worker.join();
    _jobs.erase(it);
    return true;
  }

private:
  std::string generate_id()
  {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch())
                .count();
    auto id = _next_id.fetch_add(1);
    std::stringstream ss;
    ss << std::hex << ms << "-" << std::hex << id;
    return ss.str();
  }

  std::map<std::string, std::shared_ptr<Job>> _jobs;
  std::mutex _mutex;
  std::atomic<uint64_t> _next_id;
};

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

static std::string
to_json_string(const rj::Document& doc)
{
  rj::StringBuffer buf;
  rj::Writer<rj::StringBuffer> writer(buf);
  doc.Accept(writer);
  return buf.GetString();
}

static void
add_progress_to_json(rj::Value& root,
                     rj::Document::AllocatorType& alloc,
                     const ProgressReporter& progress)
{
  rj::Value progress_obj(rj::kObjectType);

  for (auto& kv : progress.get_progress_counters()) {
    rj::Value counter_obj(rj::kObjectType);

    std::visit(
      [&](const auto& counter) {
        auto cur = static_cast<double>(counter.get_current_progress());
        auto max = static_cast<double>(counter.get_max_progress());
        double pct = (max > 0) ? (100.0 * cur / max) : 0.0;
        counter_obj.AddMember("current", cur, alloc);
        counter_obj.AddMember("max", max, alloc);
        counter_obj.AddMember("percentage", pct, alloc);
      },
      *kv.second);

    rj::Value name;
    name.SetString(kv.first.c_str(), alloc);
    progress_obj.AddMember(name, counter_obj, alloc);
  }

  root.AddMember("progress", progress_obj, alloc);
}

static std::string
format_time(const std::chrono::system_clock::time_point& tp)
{
  auto t = std::chrono::system_clock::to_time_t(tp);
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
  return ss.str();
}

static rj::Value
job_to_json(const Job& job, rj::Document::AllocatorType& alloc)
{
  rj::Value obj(rj::kObjectType);

  obj.AddMember("id", rj::Value().SetString(job.id.c_str(), alloc), alloc);
  obj.AddMember("type", rj::Value().SetString(job.type.c_str(), alloc), alloc);
  obj.AddMember("status", rj::Value().SetString(job.status.c_str(), alloc), alloc);

  obj.AddMember("created_at",
                rj::Value().SetString(format_time(job.created_at).c_str(), alloc),
                alloc);

  if (job.status != "pending") {
    obj.AddMember("started_at",
                  rj::Value().SetString(format_time(job.started_at).c_str(), alloc),
                  alloc);
  }

  if (job.status == "completed" || job.status == "failed" || job.status == "stopped") {
    obj.AddMember("finished_at",
                  rj::Value().SetString(format_time(job.finished_at).c_str(), alloc),
                  alloc);
  }

  if (job.output) {
    obj.AddMember("output",
                  rj::Value().SetString(job.output->captured().c_str(), alloc),
                  alloc);
  }

  if (!job.error_message.empty()) {
    obj.AddMember("error",
                  rj::Value().SetString(job.error_message.c_str(), alloc),
                  alloc);
  } else {
    obj.AddMember("error", rj::Value().SetNull(), alloc);
  }

  if (!job.result_summary.empty()) {
    obj.AddMember("result",
                  rj::Value().SetString(job.result_summary.c_str(), alloc),
                  alloc);
  }

  // Add progress if running
  if (job.ui_state) {
    add_progress_to_json(obj, alloc, job.ui_state->get_progress_reporter());
  }

  return obj;
}

// ---------------------------------------------------------------------------
// Argument parsing from JSON -> TilerProcess::Arguments / ConverterArguments
// ---------------------------------------------------------------------------

static std::string
get_string(const rj::Value& obj, const char* key, const std::string& default_val = "")
{
  auto it = obj.FindMember(key);
  if (it == obj.MemberEnd() || !it->value.IsString())
    return default_val;
  return it->value.GetString();
}

static int32_t
get_int(const rj::Value& obj, const char* key, int32_t default_val = 0)
{
  auto it = obj.FindMember(key);
  if (it == obj.MemberEnd() || !it->value.IsInt())
    return default_val;
  return it->value.GetInt();
}

static double
get_double(const rj::Value& obj, const char* key, double default_val = 0.0)
{
  auto it = obj.FindMember(key);
  if (it == obj.MemberEnd() || !it->value.IsNumber())
    return default_val;
  return it->value.GetDouble();
}

static bool
get_bool(const rj::Value& obj, const char* key, bool default_val = false)
{
  auto it = obj.FindMember(key);
  if (it == obj.MemberEnd() || !it->value.IsBool())
    return default_val;
  return it->value.GetBool();
}

static uint64_t
get_uint64(const rj::Value& obj, const char* key, uint64_t default_val = 0)
{
  auto it = obj.FindMember(key);
  if (it == obj.MemberEnd() || !it->value.IsUint64())
    return default_val;
  return it->value.GetUint64();
}

static std::optional<std::string>
get_optional_string(const rj::Value& obj, const char* key)
{
  auto it = obj.FindMember(key);
  if (it == obj.MemberEnd() || it->value.IsNull())
    return std::nullopt;
  if (it->value.IsString())
    return it->value.GetString();
  return std::nullopt;
}

static std::optional<uint32_t>
get_optional_uint32(const rj::Value& obj, const char* key)
{
  auto it = obj.FindMember(key);
  if (it == obj.MemberEnd() || it->value.IsNull())
    return std::nullopt;
  if (it->value.IsUint())
    return static_cast<uint32_t>(it->value.GetUint());
  return std::nullopt;
}

// Parse JSON into TilerProcess::Arguments
static tl::expected<TilerProcess::Arguments, std::string>
parse_tiler_args(const rj::Value& json)
{
  TilerProcess::Arguments args;

  // Parse sources (required)
  auto sources_it = json.FindMember("sources");
  if (sources_it == json.MemberEnd() || !sources_it->value.IsArray()) {
    return tl::make_unexpected(std::string("Missing or invalid 'sources' array"));
  }
  for (auto& src : sources_it->value.GetArray()) {
    if (src.IsString())
      args.sources.push_back(fs::path{ src.GetString() });
  }
  if (args.sources.empty()) {
    return tl::make_unexpected(std::string("'sources' array is empty"));
  }

  // Output directory
  auto outdir = get_string(json, "output_directory");
  args.output_directory = outdir.empty() ? fs::current_path() : fs::path{ outdir };

  // Spacing and diagonal fraction
  args.spacing = static_cast<float>(get_double(json, "spacing", 0.0));
  args.diagonal_fraction = get_int(json, "diagonal_fraction", 0);
  if (args.diagonal_fraction != 0) {
    args.spacing = 0;
  } else if (args.spacing == 0) {
    args.diagonal_fraction = 250;
  }

  // Point/node config
  args.max_points_per_node = static_cast<size_t>(get_uint64(json, "max_points_per_node", 20000));
  args.internal_cache_size = static_cast<size_t>(get_uint64(json, "internal_cache_size", 10000000));
  args.max_batch_read_size = static_cast<size_t>(get_uint64(json, "batch_read_size", 1000000));
  args.max_depth = static_cast<uint32_t>(get_int(json, "max_depth", 0));

  // Output format
  static const std::unordered_map<std::string, OutputFormat> format_map = {
    { "3DTILES", OutputFormat::CZM_3DTILES },
    { "BIN", OutputFormat::BIN },
    { "BINZ", OutputFormat::BINZ },
    { "LAS", OutputFormat::LAS },
    { "LAZ", OutputFormat::LAZ },
    { "ENTWINE_LAS", OutputFormat::ENTWINE_LAS },
    { "ENTWINE_LAZ", OutputFormat::ENTWINE_LAZ }
  };
  auto fmt_str = get_string(json, "output_format", "3DTILES");
  auto fmt_it = format_map.find(fmt_str);
  if (fmt_it == format_map.end()) {
    return tl::make_unexpected(std::string("Unrecognized output_format: ") + fmt_str);
  }
  args.output_format = fmt_it->second;

  // RGB mapping
  static const std::unordered_map<std::string, RGBMapping> rgb_map = {
    { "NONE", RGBMapping::None },
    { "INTENSITY_LINEAR", RGBMapping::FromIntensityLinear },
    { "INTENSITY_LOG", RGBMapping::FromIntensityLogarithmic }
  };
  auto rgb_str = get_string(json, "rgb_mapping", "NONE");
  auto rgb_it = rgb_map.find(rgb_str);
  args.rgb_mapping = (rgb_it != rgb_map.end()) ? rgb_it->second : RGBMapping::None;

  // Sampling strategy
  args.sampling_strategy = get_string(json, "sampling", "MIN_DISTANCE");

  // Source projection
  args.source_projection = get_optional_string(json, "source_projection");

  // Cache size (memory string like "800MiB")
  auto cache_size_str = get_string(json, "cache_size");
  if (!cache_size_str.empty()) {
    // Reuse parse_memory_size from executable? Not directly accessible.
    // For now, cache_size remains optional.
    // args.cache_size = ... (would need the parse function from main.cpp)
  }

  // Ignore errors
  auto ignore_str = get_string(json, "ignore_errors", "NONE");
  args.errors_to_ignore = util::IgnoreErrors::None;
  if (ignore_str != "NONE") {
    // Parse comma/space separated ignore flags
    std::stringstream ss(ignore_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
      // trim
      token.erase(0, token.find_first_not_of(" \t"));
      token.erase(token.find_last_not_of(" \t") + 1);
      if (token == "MISSING_FILES")
        args.errors_to_ignore = args.errors_to_ignore | util::IgnoreErrors::MissingFiles;
      else if (token == "INACCESSIBLE_FILES")
        args.errors_to_ignore = args.errors_to_ignore | util::IgnoreErrors::InaccessibleFiles;
      else if (token == "UNSUPPORTED_FILE_FORMAT")
        args.errors_to_ignore = args.errors_to_ignore | util::IgnoreErrors::UnsupportedFileFormat;
      else if (token == "CORRUPTED_FILES")
        args.errors_to_ignore = args.errors_to_ignore | util::IgnoreErrors::CorruptedFiles;
      else if (token == "MISSING_POINT_ATTRIBUTES")
        args.errors_to_ignore = args.errors_to_ignore | util::IgnoreErrors::MissingPointAttributes;
      else if (token == "ALL_FILE_ERRORS")
        args.errors_to_ignore = args.errors_to_ignore | util::IgnoreErrors::AllFileErrors;
      else if (token == "ALL_ERRORS")
        args.errors_to_ignore = args.errors_to_ignore | util::IgnoreErrors::AllErrors;
    }
  }

  // Tiling strategy
  auto strategy_str = get_string(json, "tiling_strategy", "FAST");
  if (strategy_str == "ACCURATE")
    args.tiling_strategy = TilingStrategy::Accurate;
  else
    args.tiling_strategy = TilingStrategy::Fast;

  // Threads
  auto threads_str = get_string(json, "threads");
  if (!threads_str.empty()) {
    // Parse "N" or "N M" format
    std::stringstream ss(threads_str);
    std::vector<std::string> tokens;
    std::string token;
    while (ss >> token)
      tokens.push_back(token);
    if (tokens.size() == 1) {
      try {
        auto n = static_cast<uint32_t>(std::stoul(tokens[0]));
        args.thread_config = ThreadConfig{ AdaptiveThreadCount{ n } };
      } catch (...) {}
    } else if (tokens.size() >= 2) {
      try {
        auto read_n = static_cast<uint32_t>(std::stoul(tokens[0]));
        auto idx_n = static_cast<uint32_t>(std::stoul(tokens[1]));
        args.thread_config = ThreadConfig{ FixedThreadCount{ read_n, idx_n } };
      } catch (...) {}
    }
  } else {
    auto n = std::thread::hardware_concurrency();
    args.thread_config = ThreadConfig{ AdaptiveThreadCount{ n } };
  }

  return args;
}

// Parse JSON into ConverterArguments
static tl::expected<ConverterArguments, std::string>
parse_converter_args(const rj::Value& json)
{
  ConverterArguments args;

  auto src = get_string(json, "source_folder");
  if (src.empty()) {
    return tl::make_unexpected(std::string("Missing 'source_folder'"));
  }
  args.source_folder = src;

  auto out = get_string(json, "output_folder");
  args.output_folder = out.empty() ? fs::current_path().string() : out;

  static const std::unordered_map<std::string, OutputFormat> format_map = {
    { "3DTILES", OutputFormat::CZM_3DTILES },
    { "LAS", OutputFormat::LAS },
    { "LAZ", OutputFormat::LAZ }
  };
  auto fmt_str = get_string(json, "output_format", "3DTILES");
  auto fmt_it = format_map.find(fmt_str);
  if (fmt_it == format_map.end()) {
    return tl::make_unexpected(std::string("Unrecognized output_format: ") + fmt_str);
  }
  args.output_format = fmt_it->second;

  args.source_projection = get_optional_string(json, "source_projection");
  args.max_depth = get_optional_uint32(json, "max_depth");
  args.delete_source_files = get_bool(json, "delete_source", false);

  return args;
}

// ---------------------------------------------------------------------------
// HTTP route handlers
// ---------------------------------------------------------------------------

static void
handle_health(const httplib::Request& req, httplib::Response& res)
{
  rj::Document doc;
  doc.SetObject();
  doc.AddMember("status", "ok", doc.GetAllocator());
  res.set_content(to_json_string(doc), "application/json");
}

static void
handle_create_job(JobManager& manager, const httplib::Request& req, httplib::Response& res)
{
  rj::Document body;
  body.Parse(req.body.c_str());

  if (body.HasParseError()) {
    res.status = 400;
    res.set_content("{\"error\":\"Invalid JSON\"}", "application/json");
    return;
  }

  if (!body.HasMember("type") || !body["type"].IsString()) {
    res.status = 400;
    res.set_content("{\"error\":\"Missing 'type' field (tile or convert)\"}", "application/json");
    return;
  }

  std::string type = body["type"].GetString();

  if (type == "tile") {
    auto args_result = parse_tiler_args(body);
    if (!args_result) {
      res.status = 400;
      rj::Document err;
      err.SetObject();
      err.AddMember("error",
                    rj::Value().SetString(args_result.error().c_str(), err.GetAllocator()),
                    err.GetAllocator());
      res.set_content(to_json_string(err), "application/json");
      return;
    }

    auto args = std::move(*args_result);
    auto job_id = manager.create_job("tile", [args = std::move(args)](Job& job) mutable {
      // Wire stop_source and external progress reporter through args
      args.stop_source = &job.stop_requested;
      args.external_progress_reporter = &job.ui_state->get_progress_reporter();

      // Set up journal config
      global_config().is_journaling_enabled = false;
      global_config().root_directory = args.output_directory;

      TilerProcess tiler{ args };
      tiler.run();

      std::stringstream ss;
      ss << "Tiling completed. Output: " << args.output_directory.string();
      job.result_summary = ss.str();
    });

    rj::Document resp;
    resp.SetObject();
    resp.AddMember("id", rj::Value().SetString(job_id.c_str(), resp.GetAllocator()), resp.GetAllocator());
    resp.AddMember("status", "pending", resp.GetAllocator());
    res.set_content(to_json_string(resp), "application/json");

  } else if (type == "convert") {
    auto args_result = parse_converter_args(body);
    if (!args_result) {
      res.status = 400;
      rj::Document err;
      err.SetObject();
      err.AddMember("error",
                    rj::Value().SetString(args_result.error().c_str(), err.GetAllocator()),
                    err.GetAllocator());
      res.set_content(to_json_string(err), "application/json");
      return;
    }

    auto args = std::move(*args_result);
    auto job_id = manager.create_job("convert", [args = std::move(args)](Job& job) mutable {
      args.stop_source = &job.stop_requested;
      args.external_progress_reporter = &job.ui_state->get_progress_reporter();

      try {
        run_conversion(args);
        if (job.stop_requested) {
          job.status = "stopped";
          job.result_summary = "Conversion stopped by user request";
          return;
        }
        std::stringstream ss;
        ss << "Conversion completed. Output: " << args.output_folder;
        job.result_summary = ss.str();
      } catch (const std::exception& ex) {
        if (job.stop_requested) {
          job.result_summary = "Conversion stopped by user request";
          return;
        }
        throw;
      }
    });

    rj::Document resp;
    resp.SetObject();
    resp.AddMember("id", rj::Value().SetString(job_id.c_str(), resp.GetAllocator()), resp.GetAllocator());
    resp.AddMember("status", "pending", resp.GetAllocator());
    res.set_content(to_json_string(resp), "application/json");

  } else {
    res.status = 400;
    res.set_content("{\"error\":\"type must be 'tile' or 'convert'\"}", "application/json");
  }
}

static void
handle_list_jobs(JobManager& manager, const httplib::Request& req, httplib::Response& res)
{
  auto jobs = manager.list_jobs();

  rj::Document doc;
  doc.SetObject();
  auto& alloc = doc.GetAllocator();

  rj::Value jobs_arr(rj::kArrayType);
  for (auto& job : jobs) {
    rj::Value job_obj(rj::kObjectType);
    job_obj.AddMember("id", rj::Value().SetString(job->id.c_str(), alloc), alloc);
    job_obj.AddMember("type", rj::Value().SetString(job->type.c_str(), alloc), alloc);
    job_obj.AddMember("status", rj::Value().SetString(job->status.c_str(), alloc), alloc);
    job_obj.AddMember("created_at",
                      rj::Value().SetString(format_time(job->created_at).c_str(), alloc),
                      alloc);
    jobs_arr.PushBack(job_obj, alloc);
  }

  doc.AddMember("jobs", jobs_arr, alloc);
  res.set_content(to_json_string(doc), "application/json");
}

static void
handle_get_job(JobManager& manager, const httplib::Request& req, httplib::Response& res)
{
  auto job_id = req.matches[1];
  auto job = manager.get_job(job_id);

  if (!job) {
    res.status = 404;
    res.set_content("{\"error\":\"Job not found\"}", "application/json");
    return;
  }

  rj::Document doc;
  doc.SetObject();
  auto job_json = job_to_json(*job, doc.GetAllocator());

  // Copy all members from job_json to doc
  for (auto it = job_json.MemberBegin(); it != job_json.MemberEnd(); ++it) {
    rj::Value key;
    key.CopyFrom(it->name, doc.GetAllocator());
    rj::Value val;
    val.CopyFrom(it->value, doc.GetAllocator());
    doc.AddMember(key, val, doc.GetAllocator());
  }

  res.set_content(to_json_string(doc), "application/json");
}

static void
handle_stop_job(JobManager& manager, const httplib::Request& req, httplib::Response& res)
{
  auto job_id = req.matches[1];

  if (!manager.stop_job(job_id)) {
    res.status = 404;
    res.set_content("{\"error\":\"Job not found or not running\"}", "application/json");
    return;
  }

  rj::Document doc;
  doc.SetObject();
  auto job_id_str = job_id.str();
  doc.AddMember("id", rj::Value().SetString(job_id_str.c_str(), doc.GetAllocator()), doc.GetAllocator());
  doc.AddMember("status", "stopping", doc.GetAllocator());
  res.set_content(to_json_string(doc), "application/json");
}

static void
handle_delete_job(JobManager& manager, const httplib::Request& req, httplib::Response& res)
{
  auto job_id = req.matches[1];

  if (!manager.remove_job(job_id)) {
    res.status = 404;
    res.set_content("{\"error\":\"Job not found or still running\"}", "application/json");
    return;
  }

  rj::Document doc;
  doc.SetObject();
  doc.AddMember("deleted", true, doc.GetAllocator());
  res.set_content(to_json_string(doc), "application/json");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int
main(int argc, char** argv)
{
  int port = 8080;
  if (argc >= 2) {
    try {
      port = std::stoi(argv[1]);
    } catch (...) {
      std::cerr << "Usage: " << argv[0] << " [port]" << std::endl;
      return 1;
    }
  }

  JobManager manager;

  httplib::Server server;

  // Health check
  server.Get("/api/health", handle_health);

  // Job endpoints
  server.Post("/api/jobs", [&](const httplib::Request& req, httplib::Response& res) {
    handle_create_job(manager, req, res);
  });

  server.Get("/api/jobs", [&](const httplib::Request& req, httplib::Response& res) {
    handle_list_jobs(manager, req, res);
  });

  server.Get(R"(/api/jobs/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
    handle_get_job(manager, req, res);
  });

  server.Post(R"(/api/jobs/([^/]+)/stop)", [&](const httplib::Request& req, httplib::Response& res) {
    handle_stop_job(manager, req, res);
  });

  server.Delete(R"(/api/jobs/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
    handle_delete_job(manager, req, res);
  });

  std::cout << "Schwarzwald HTTP Service listening on port " << port << std::endl;
  std::cout << "Endpoints:" << std::endl;
  std::cout << "  GET  /api/health" << std::endl;
  std::cout << "  POST /api/jobs" << std::endl;
  std::cout << "  GET  /api/jobs" << std::endl;
  std::cout << "  GET  /api/jobs/:id" << std::endl;
  std::cout << "  POST /api/jobs/:id/stop" << std::endl;
  std::cout << "  DELETE /api/jobs/:id" << std::endl;

  if (!server.listen("0.0.0.0", port)) {
    std::cerr << "Failed to start server on port " << port << std::endl;
    return 1;
  }

  return 0;
}
