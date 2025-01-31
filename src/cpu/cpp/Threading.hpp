/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include <vector>

namespace zentorch {

/**
 * @brief Accepts a list of cores and binds the current process
 * to the gives cores
 *
 * @param cpu_core_list Lists of cores for the process to be binded
 */
void thread_bind(const std::vector<int32_t> &cpu_core_list);

} // namespace zentorch
