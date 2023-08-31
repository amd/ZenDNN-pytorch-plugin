Copyright &copy; 2023 Advanced Micro Devices, Inc. All rights reserved.

This file details the technical contributions made to zentorch. If you are interested in contributing to zentorch, please read through this!

# Table of Contents

<!-- toc -->
- [Making changes to zentorch](#making-changes-to-zentorch)
- [Codebase structure](#codebase-structure)
- [Coding-style guidelines](#coding-style-guidelines)
  - [Linting mechanism](#linting-mechanism)
- [License header check](#license-header-check)
- [Logging and Profiling](#logging-and-profiling)
- [Unit-testing](#unit-testing)
- [Git commit guidelines](#git-commit-guidelines)
<!-- tocstop -->

## Making changes to zentorch
You will have to install zentorch from [source](README.md#from-source) to start contributing. You should run the [linting checks](#linting-mechanism), [license header check](#license-header-check) and [unit-tests](#unit-testing) before creating a PR. Once you have the changes ready, create a PR and follow the instructions given under the section [Git commit guidelines](#git-commit-guidelines).

## Codebase structure
* [cmake](cmake) - Downloads and builds AOCL BLIS and ZenDNN in [third_party](third_party) directory. For more details refer to [FindZENDNN.cmake](cmake/modules/FindZENDNN.cmake).
* [src](src/cpu) - Contains python and cpp sources for zentorch.
* [linter](linter) - Shell script for linting is present in this directory.
* [test](test) - Python based unit-tests for zentorch functionality.
* [setup.py](setup.py) - Wheel file build script using setuptools.
* [build.sh](build.sh) - Lightweight shell script for building, using [setup.py](setup.py).
* [license_header_check.py](license_header_check.py) - Checks for the presence of AMD copyright header.

## Coding-style guidelines
zentorch follows the **PEP-8** guidelines for Python and the **LLVM** style for C/C++ code. [Linting mechanism](#linting-mechanism) section below gives further details.

### Linting mechanism
You can perform a code-check on all Python and CPP files by running the following command from repo root.
```bash
bash linter/py_cpp_linter.sh
```
This will install all the prerequisites and then perform code check; the script displays the optional commands to re-format as well. The repo uses a combination of **flake8** and **black** for linting and formatting the python files and **clang-format** for the C/C++ sources.

## License header check
To check for the presence of license headers, we have a comment style agnostic python script, which can be invoked as given below from the repo root.
```bash
python license_header_check.py
```
For example, the license header for a .cpp file is:
```cpp
/******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
```

## Logging and Profiling
Logging is disabled by default but can be enabled by using the environment variable **ZENDNN_LOG_OPTS** before running any tests. Its behavior can be specified by setting **ZENDNN_LOG_OPTS** to a comma-delimited list of **ACTOR:DBGLVL** pairs. An example to turn on info logging is given below.
```bash
export ZENDNN_LOG_OPTS=ALL:2
```
To enable the profiling logs **zendnn_primitive_create** and **zendnn_primitive_execute**, you can use:
```bash
export ZENDNN_PRIMITIVE_LOG_ENABLE=1
```

For further details on logging, refer to ZenDNN user-guide from [this page](https://www.amd.com/en/developer/zendnn.html).

## Unit-testing
Unit tests for Python are located in a script test_zentorch.py inside the test directory. It contains tests for all ops supported by zentorch, bf16 device support check and a few other tests. The pre-requisites for running or adding new tests are the **expecttest** and **hypothesis** packages. To run the tests:
```bash
python test/test_zentorch.py
```

## Git commit guidelines
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
