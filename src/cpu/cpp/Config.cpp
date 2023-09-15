/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "Config.hpp"

#include <blis.h>
#include <zendnn_version.h>

#include <sstream>

namespace ZenDNNTorch {

std::string get_zendnn_version() {
  std::ostringstream ss;
  ss << ZENDNN_VERSION_MAJOR << "." << ZENDNN_VERSION_MINOR << "."
     << ZENDNN_VERSION_PATCH;
  return ss.str();
}

std::string show_config() {
  std::ostringstream ss;
  ss << "torch_zendnn_plugin built with:\n";
  ss << "  - "
     << "AMD " << bli_info_get_version_str() << " ( Git Hash "
     << BLIS_VERSION_HASH << " )"
     << "\n";
  ss << "  - "
     << "AMD ZENDNN v" << get_zendnn_version() << " ( Git Hash "
     << ZENDNN_LIB_VERSION_HASH << " )"
     << "\n";

  return ss.str();
}

} // namespace ZenDNNTorch
