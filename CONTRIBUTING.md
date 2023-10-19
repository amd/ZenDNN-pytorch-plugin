Copyright &copy; 2023 Advanced Micro Devices, Inc. All rights reserved.

This file details the technical contributions made to zentorch. If you are interested in contributing to zentorch, please read through this!

Table of Contents
============

<!-- toc -->
- [Making changes to zentorch](#1-making-changes-to-zentorch)
- [Codebase structure](#2-codebase-structure)
- [Coding-style guidelines](#3-coding-style-guidelines)
  - [Linting mechanism](#31-linting-mechanism)
- [Adding log messages to sources](#4-adding-log-messages-to-sources)
- [Unit-testing](#5-unit-testing)
- [Git commit guidelines](#6-git-commit-guidelines)
<!-- tocstop -->

# 1. Making changes to zentorch
You will have to install zentorch from [source](README.md#from-source) to start contributing. You should run the [linting checks](#linting-mechanism), [license header check](#license-header-check) and [unit-tests](#unit-testing) before creating a PR. Once you have the changes ready, create a PR and follow the instructions given under the section [Git commit guidelines](#git-commit-guidelines).

# 2. Codebase structure
* [cmake](cmake) - Downloads and builds AOCL BLIS and ZenDNN in [third_party](third_party) directory. For more details refer to [FindZENDNN.cmake](cmake/modules/FindZENDNN.cmake).
* [src](src/cpu) - Contains python and cpp sources for zentorch.
* [linter](linter) - Shell script for linting is present in this directory.
* [test](test) - Python based unit-tests for zentorch functionality.
* [setup.py](setup.py) - Wheel file build script using setuptools.
* [build.sh](build.sh) - Lightweight shell script for building, using [setup.py](setup.py).
* [license_header_check.py](license_header_check.py) - Checks for the presence of AMD copyright header.

# 3. Coding-style guidelines
zentorch follows the **PEP-8** guidelines for Python and the **LLVM** style for C/C++ code. [Linting mechanism](#linting-mechanism) section below gives further details.

## 3.1. Linting mechanism
You can perform a code-check on all Python and CPP files by running the following command from repo root.
```bash
bash linter/py_cpp_linter.sh
```
This will install all the prerequisites and then perform code check; the script displays the optional commands to re-format as well. The repo uses a combination of **flake8** and **black** for linting and formatting the python files and **clang-format** for the C/C++ sources.

>**IMPORTANT**: Since the script uses git integration for clang-format, it should be run after the files have been added to the staging area i.e., files untracked by git are not checked for CPP coding style and will not be modified. First, `git add` the files and then run the linter script.

# 4. Adding log messages to sources
It is a good practice to add logging messages along with the changes you make.

For CPP source files, zentorch supports the LOG() macro. An example to put an error message is given below:
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

# 5. Unit-testing
Unit tests for Python are located in a script test_zentorch.py inside the test directory. It contains tests for all ops supported by zentorch, bf16 device support check and a few other tests. The pre-requisites for running or adding new tests are the **expecttest** and **hypothesis** packages. To run the tests:
```bash
python test/test_zentorch.py
```

# 6. Git commit guidelines
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
