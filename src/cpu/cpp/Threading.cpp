/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <omp.h>
#include <stdexcept>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#include "Threading.hpp"

namespace zentorch {

void thread_bind(const std::vector<int32_t> &cpu_core_list) {
  omp_set_num_threads(cpu_core_list.size());

#pragma omp parallel num_threads(cpu_core_list.size())
  {
    int thread_index = omp_get_thread_num();
#ifdef _WIN32
    DWORD_PTR mask = static_cast<DWORD_PTR>(1)
                     << cpu_core_list[thread_index];
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
      throw std::runtime_error("Fail to bind cores.");
    }
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core_list[thread_index], &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) !=
        0) {
      throw std::runtime_error("Fail to bind cores.");
    }
#endif
  }
}

} // namespace zentorch
