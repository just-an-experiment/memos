# pure-function-with-tests-so4o

This module generates pure functions based on user queries, verifies their purity, and creates corresponding tests.

## Features

- Generate pure functions in multiple programming languages
- Verify function purity for Python functions
- Create comprehensive tests for generated functions
- Save generated functions and tests to files
- Configurable maximum attempts for generation
- Configurable number of tests to generate (default: 3)

## Usage

Run the script and follow the prompts:

1. Enter your query to generate a pure function
2. Choose a programming language (default: Python)
3. Set maximum attempts for generation (default: 3)
4. Set number of tests to generate (default: 3)

The generated function and tests will be displayed and saved if the function is pure.

## Module Export

The main module can be imported to:

1. Generate pure functions
2. Verify function purity
3. Generate tests for functions
4. Save generated functions and tests

## Example


```python
from pure_function_with_tests_so4o import main as pure_function_with_tests_so4o_main, ProgrammingLanguage

# Generate and save a pure function from query
generated_function = pure_function_with_tests_so4o_main(
    "your query here",
    language=ProgrammingLanguage.PYTHON,
    max_attempts=3,
    num_tests=3
)
```
