# About Tests

## Overview

All the tests have been divided into the following groups

- **unittests** :- This folder consists of all the unit tests for Zentorch.
  To run these tests use

  ```bash
   python -m unittest discover -s ./test/unittests
  ```

  - **_model_tests_** :- These are the unit tests consisting of Custom Models.
  - **_op_tests_** :- These are the unit tests for individual ops or multiple ops not in form of Custom Models.
  - **_miscellaneous_tests_** :- These are the unit tests for miscellaneous(non-op) things.

- **llm_tests** :- These are the tests for major operators used in LLMs
  To run these tests use

  ```bash
   python -m unittest discover -s ./test/llm_tests
  ```

- **pre_trained_models** :- These are tests for pre-trained models constisting of multiple operators.
  To run these tests use

  ```bash
   python -m unittest discover -s ./test/pre_trained_model_tests
  ```

## Testing Guide

Follow the below steps to run the tests.

- To run all tests.

  ```bash
  python -m unittest discover -s ./test
  ```

- To run all tests in a particular folder use.

  ```bash
  python -m unittest discover -s ./test/unittests
  ```

- To run tests in a particular file use.

  ```bash
  python -m unittest test/unittests/op_tests/test_bmm.py
  ```

To filter out tests in a subset use.

- `-k "<pattern>"` to filter based on test names(file_name+class_name+function_name).

  Example: To run all test with "woq" in their name please use below command.

  ```bash
  python -m unittest discover -s ./test/unittests -k "woq"
  ```

- `-p "<pattern>"` to filter based on file names.

  Example: To run tests in all the files with "test_mm" in their file name please use below command.

  ```bash
  python -m unittest discover -s ./test/unittests -p "test_mm*"
  ```
