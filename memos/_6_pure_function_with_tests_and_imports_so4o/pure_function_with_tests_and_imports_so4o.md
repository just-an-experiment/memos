# pure-function-with-tests-and-imports-so4o

Given a set of folders and a query, this module generates a file with a pure function in python. The function fulfills the query. The file imports the relevant functions provided in the folder files.

# Features
- Generate a pure function spec in python
- Generate a pure function implementation in python
- Optionally generate tests for the function
- Generate relevant imports for the function based on the extracted file and function tree

# Example
```python
from pure_function_with_tests_and_imports_so4o import main as pure_function_with_tests_and_imports_so4o_main

# Generate a pure function with tests and imports
result = pure_function_with_tests_and_imports_so4o_main("Create a function to calculate the sum of two numbers")
```

# Notes
- The generated function is pure ONLY if the imported functions are pure.