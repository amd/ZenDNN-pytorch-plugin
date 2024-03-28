Copyright &copy; 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

This file details the technical contributions made to _zentorch_. If you are interested in contributing to _zentorch_, please read through this!

Table of Contents
============

<!-- toc -->
- [Making changes to _zentorch_](#1-making-changes-to-zentorch)
- [Codebase structure](#2-codebase-structure)
- [Adding a custom Op via _zentorch_](#3-adding-a-custom-op-via-zentorch)
  - [Implementation of the function for custom op](#31-implementation-of-the-function-for-custom-op)
  - [Declaration in ZenTorchOps.hpp](#32-declaration-in-zentorchopshpp)
  - [Registration of the op with TORCH_LIBRARY and TORCH_LIBRARY_IMPL](#33-registration-of-the-op-with-torch_library-and-torch_library_impl)
  - [Registration of fake tensor functions](#34-registration-of-fake-tensor-functions)
  - [General Guidelines](#35-general-guidelines)
- [Coding-style guidelines](#4-coding-style-guidelines)
  - [Linting mechanism](#41-linting-mechanism)
- [Adding log messages to sources](#5-adding-log-messages-to-sources)
- [Unit-testing](#6-unit-testing)
- [Git commit guidelines](#7-git-commit-guidelines)
<!-- tocstop -->

# 1. Making changes to _zentorch_
You will have to install _zentorch_ from [source](README.md#from-source) to start contributing. You should run the [linting checks](#linting-mechanism), [license header check](#license-header-check) and [unit-tests](#unit-testing) before creating a PR. Once you have the changes ready, create a PR and follow the instructions given under the section [Git commit guidelines](#git-commit-guidelines).

# 2. Codebase structure
* [cmake](cmake) - Downloads and builds AOCL BLIS and ZenDNN in [third_party](third_party) directory. For more details refer to [FindZENDNN.cmake](cmake/modules/FindZENDNN.cmake).
* [src](src/cpu) - Contains python and cpp sources for _zentorch_.
* [linter](linter) - Shell script for linting is present in this directory.
* [test](test) - Python based unit-tests for _zentorch_ functionality.
* [setup.py](setup.py) - Wheel file build script using setuptools.
* [build.sh](build.sh) - Lightweight shell script for building, using [setup.py](setup.py).
* [license_header_check.py](license_header_check.py) - Checks for the presence of AMD copyright header.

# 3. Adding a custom Op via _zentorch_
An op at cpp level in _zentorch_ acts as wrapper/bridge between torch data structures and their respective APIs and similarly between zendnn data structures and the corresponding APIs. Whenever we need to add a new op we need to follow the steps given below.
## 3.1. Implementation of the function for custom op
The actual implementation of the op can be written in any existing cpp files corresponding to the op or a new cpp file dedicated to the new op.
```cpp
return_type zendnn_op_impl(const at::Tensor &tensor_parameter,
                           const bool &boolean_parameter,
                           const int64_t &int_parameter,
                           ...) {

  /*
    logic for the new zendnn_op
  */

}
```
## 3.2. Declaration in ZenTorchOps.hpp
The corresponding C++ function protoype must also be added in the file `src/cpu/cpp/ZenTorchOps.hpp`. This must be inside the `zentorch` namespace.
```cpp
  return_type zendnn_op_impl(const at::Tensor &tensor_parameter,
                             const bool &boolean_parameter,
                             const int64_t &int_parameter,
                             ...);
```
## 3.3. Registration of the op with TORCH_LIBRARY and TORCH_LIBRARY_IMPL
Whenever we need to add a new op we need to add the prototype of the new op function as an entry to TORCH_LIBRARY in `src/cpu/cpp/Bindings.cpp`.
The new op implementation must be registered with the corresponding name intended to be used from the python framework, as follows. The function prototype should follow `aten/src/ATen/native/README.md` in PyTorch repo.
```cpp
  TORCH_LIBRARY(zentorch, m) {
    m.def("zendnn_op(Tensor tensor_parameter, "
          "bool boolean_parameter, int int_parameter, ..."
          ") -> return_value");

    /*
    other ops
    */
  }
```

Register the implementation corresponding the above op with TORCH_LIBRARY_IMPL as follows.
```cpp
  TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
    m.impl("zendnn_op", zentorch::zendnn_op_impl);

    /*
    other ops
    */
}
```
Here the "zendnn_op" will be available on the python side via `torch.ops.zentorch.zendnn_op`. The zendnn_op_impl must be present in any existing cpp files corresponding to the op or a new cpp file.

>Note:
Following are guidelines to consider when creating and registering a new op.
>  - If there is simlar op in ATen of PyTorch, please check "aten/src/ATen/native/native_functions.yaml" in PyTorch repo.
>  - Our op arguments should be superset of the corresponding arguments in ATen op.
>  - Our op arguments should match the arguments of corresponding op in both order of the arguments and type.
>  - Our op specific arguments should be at the end of the list.
>  - All ops should have prefix "zendnn_", for example zendnn_op.
>  - Add a corresponding unit test for the new op created in `test/test_zentorch.py` being in line with the other tests. For additional details refer to [Unit-testing](#6-unit-testing)

## 3.4. Registration of fake tensor functions
The op also must be registered in the `src/cpu/python/zentorch/_meta_registrations.py` with the decorator @register_meta("{op_name}"). This registration has the function protoype and the corresponding output is returned with the appropriate shapes. This registration happens in pythonic way.
```python
@register_meta("{zendnn_op}")
def meta_zendnn_op(
    ...parameters
):
    output = ...

    # logic to calculate the output shape
    # using the parameters

    return output

make_fallback(torch.ops.zentorch.zendnn_op)
```

# 4. Coding-style guidelines
_zentorch_ follows the **PEP-8** guidelines for Python and the **LLVM** style for C/C++ code. [Linting mechanism](#linting-mechanism) section below gives further details.

## 4.1. Linting mechanism
You can perform a code-check on all Python and CPP files by running the following command from repo root.
```bash
bash linter/py_cpp_linter.sh
```
This will install all the prerequisites and then perform code check; the script displays the optional commands to re-format as well. The repo uses a combination of **flake8** and **black** for linting and formatting the python files and **clang-format** for the C/C++ sources.

>**IMPORTANT**: Since the script uses git integration for clang-format, it should be run after the files have been added to the staging area i.e., files untracked by git are not checked for CPP coding style and will not be modified. First, `git add` the files and then run the linter script.

# 5. Adding log messages to sources
It is a good practice to add logging messages along with the changes you make.

For CPP source files, _zentorch_ supports the LOG() macro. An example to put an error message is given below:
```cpp
LOG(ERROR) << "This is an error message!";
```
For Python sources, first import the custom logging module, set the logger for the file and then put the logging messages in your code:
```python
from ._logging import get_logger
logger = get_logger(__name__)
--snip--
logger.info("This is an info message!")
--snip--
```
The log levels have been discussed [here](README.md#42-zentorch-logs).

# 6. Unit-testing
Unit tests for Python are located in a script `test_zentorch.py` inside the test directory. It contains tests for all ops supported by _zentorch_, bf16 device support check and a few other tests. The pre-requisites for running or adding new tests are the **expecttest** and **hypothesis** packages. To run the tests:
```bash
python test/test_zentorch.py
```

# 7. Git commit guidelines
Don't use `git commit -m <your message>` option as you cannot compose the body of the git commit message with this, instead use `git commit -s` to add a sign-off and be more descriptive about your change.

Use module names at the beginning of your commit message, an example for ZENTORCH CORE is given below:
```
[ZENTORCH CORE] This is the subject of commit

  - this is the body of commit message
```
The module names are:<br>
**ZENTORCH CORE** - all changes to cpp and python integration code.<br>
**ZENTORCH PTNR** - changes to partitioner.<br>
**ZENTORCH INFRA** - build system and other infrastructural changes.<br>
**ZENTORCH TEST** - changes to test and validation scripts.

Additionally, it is recommended to follow the seven rules of a great git commit message:
* Separate subject from body with a blank line.
* Limit the subject line to 50 characters.
* Capitalize the subject line.
* Do not end the subject line with a period.
* Use the imperative mood in the subject line.
* Wrap the body at 72 characters.
* Use the body to explain what and why vs how.
