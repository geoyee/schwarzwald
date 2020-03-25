#pragma once

#include "datastructures/PointBuffer.h"
#include "math/AABB.h"
#include "octree/MortonIndex.h"
#include "pointcloud/PointAttributes.h"

#include <mutex>
#include <unordered_map>

/**
 * Persists points in memory
 */
struct MemoryPersistence {
  MemoryPersistence();
  MemoryPersistence(const MemoryPersistence &) = delete;
  MemoryPersistence(MemoryPersistence &&) = default;
  MemoryPersistence &operator=(const MemoryPersistence &) = delete;
  MemoryPersistence &operator=(MemoryPersistence &&) = default;

  template <typename Iter>
  void persist_points(Iter points_begin, Iter points_end, const AABB &bounds,
                      const std::string &node_name) {
    std::lock_guard<std::mutex> lock{*_lock};
    auto &buffer = _points_cache[node_name];
    std::for_each(points_begin, points_end, [&buffer](const auto &point_ref) {
      buffer.push_point(point_ref);
    });
  }

  void persist_points(PointBuffer const &points, const AABB &bounds,
                      const std::string &node_name);

  void retrieve_points(const std::string &node_name, PointBuffer &points);

  const auto &get_points() const { return _points_cache; }

private:
  std::unique_ptr<std::mutex> _lock;
  std::unordered_map<std::string, PointBuffer> _points_cache;
};