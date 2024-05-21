/******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "Config.hpp"

#include <blis.h>
#include <zendnn_version.h>

#include <sstream>

namespace zentorch {

#define TO_STRING2(x) #x
#define TO_STRING(x) TO_STRING2(x)

std::string get_zendnn_version() {
  std::ostringstream ss;
  ss << ZENDNN_VERSION_MAJOR << "." << ZENDNN_VERSION_MINOR << "."
     << ZENDNN_VERSION_PATCH;
  return ss.str();
}

std::string show_config() {
  std::ostringstream ss;
  ss << "zentorch Version: " << TO_STRING(ZENTORCH_VERSION) << "\n";
  ss << "zentorch built with:\n";
  ss << "  - Commit-id: " << TO_STRING(ZENTORCH_VERSION_HASH) << "\n";
  ss << "  - PyTorch: " << TO_STRING(PT_VERSION) << "\n";
#if defined(__GNUC__)
  ss << "  - GCC Version: " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
#endif
#if defined(__cplusplus)
  ss << "  - C++ Version: " << __cplusplus << "\n";
#endif
  ss << "Third_party libraries:\n";
  ss << "  - "
     << "AMD " << bli_info_get_version_str() << " ( Git Hash "
     << BLIS_VERSION_HASH << " )"
     << "\n";
  ss << "  - "
     << "AMD ZENDNN v" << get_zendnn_version() << " ( Git Hash "
     << ZENDNN_LIB_VERSION_HASH << " )"
     << "\n";
  ss << "  - "
     << "FBGEMM " << FBGEMM_VERSION_TAG << " ( Git Hash " << FBGEMM_VERSION_HASH
     << " )"
     << "\n";

  return ss.str();
}

} // namespace zentorch
