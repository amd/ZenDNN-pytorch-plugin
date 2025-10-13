/******************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include "Utils.hpp"
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace zentorch {
class EnvReader {
public:
  static EnvReader &getInstance() {
    static EnvReader instance;
    return instance;
  }

  static void initializeEnvVariables() {
    // std::call_once guarantees that the lambda is executed exactly once
    // across all threads, no matter how many times this is called.
    std::call_once(getInstance().initFlag,
                   [] { getInstance().initializeVariables(); });
  }

  static int getEnvVariableAsInt(const std::string_view &varName) {
    // We still need to ensure initialization has run before any "get" call.
    // This call is now extremely cheap after the first time.
    initializeEnvVariables();
    return getInstance().getStoredVariableAsInt(varName);
  }

private:
  static inline std::unordered_map<std::string_view, int> envVariables;
  mutable std::once_flag initFlag; // Replaces the bool flag

  EnvReader() {}

  void initializeVariables() {
    storeEnvVariable("USE_ZENDNN_MATMUL_DIRECT", 0);
    storeEnvVariable("USE_ZENDNN_SDPA_MATMUL_DIRECT", 0);
    storeEnvVariable("ZENDNN_ZENDNNL", 1); // ZenDNNL is used by default
  }

  // Function to convert and store environment variable value as integer
  void storeEnvVariable(const std::string_view &varName_view,
                        const int &default_value) {
    std::string varName = std::string(varName_view);
    const char *env_value = std::getenv(varName.c_str());
    if (env_value) {
      try {
        int env_value_int = std::stoi(env_value);
        if (env_value_int != 0 && env_value_int != 1) {
          LOG(WARNING) << "Environment value of: " << "'" << varName_view << "'"
                       << " is not one of allowed values (0 or 1). Execution "
                       << "will use the default value of " << default_value
                       << "\n";
          envVariables[varName_view] = default_value;
        } else {
          envVariables[varName_view] = env_value_int;
        }
      } catch (const std::invalid_argument &e) {
        LOG(WARNING)
            << "Value of environment variable '" << varName_view << "'"
            << " could not be converted into a compatible format and"
            << " is invalid. \nExecution will use the default value of "
            << default_value << "\n";
        envVariables[varName_view] = default_value;
      } catch (const std::out_of_range &e) {
        LOG(WARNING)
            << "Value of environment variable '" << varName_view << "'"
            << " is out of range and cannot be converted to a compatible"
            << " format. \nExecution will use the default value of "
            << default_value << "\n";
        envVariables[varName_view] = default_value;
      }
    } else {
      envVariables[varName_view] = default_value;
    }
  }

  // Function to retrieve stored environment variable as integer
  int getStoredVariableAsInt(const std::string_view &varName) {
    auto it = envVariables.find(varName);
    ZENTORCH_CHECK(it != envVariables.end(), "Variable ", varName, " not found")
    return it->second;
  }

  // Prevent copying and assignment
  EnvReader(const EnvReader &) = delete;
  EnvReader &operator=(const EnvReader &) = delete;
};

} // namespace zentorch
