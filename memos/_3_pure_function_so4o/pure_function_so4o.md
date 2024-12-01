# pure-function-so4o

This module generates pure functions based on user queries and verifies their purity.

## Features

- Generate pure functions in multiple programming languages
- Verify function purity for Python functions
- Save generated functions to files
- Configurable maximum attempts for generation

## Usage

Run the script and follow the prompts:

1. Enter your query to generate a pure function
2. Choose a programming language (default: Python)
3. Set maximum attempts for generation (default: 3)

The generated function will be displayed and saved if it's pure.

## Module Export

The main module can be imported to:

1. Generate pure functions
2. Verify function purity
3. Save generated functions

## Example

```python
from pure_function_so4o import main as pure_function_so4o_main, ProgrammingLanguage

# Generate and save a pure function from query
generated_function = pure_function_so4o_main(
    "your query here",
    language=ProgrammingLanguage.PYTHON,
    max_attempts=3
)
```
