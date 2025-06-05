/******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <vector>

namespace zentorch {

class DataPointerManager {
public:
  static DataPointerManager &getInstance() {
    static DataPointerManager instance;
    return instance;
  }

  void addPointer(uintptr_t ptr) { pointers.push_back(ptr); }

  const std::vector<uintptr_t> &getPointers() const { return pointers; }

  void clear() {
    pointers.clear();
    pointers.shrink_to_fit();
  }

private:
  DataPointerManager() = default;
  std::vector<uintptr_t> pointers;

  // Delete copy/move constructors and assignment operators
  DataPointerManager(const DataPointerManager &) = delete;
  DataPointerManager &operator=(const DataPointerManager &) = delete;
  DataPointerManager(DataPointerManager &&) = delete;
  DataPointerManager &operator=(DataPointerManager &&) = delete;
};

} // namespace zentorch
