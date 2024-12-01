# structured-output-4o

This module generates pydantic models that may be passed into GPT-4's structured output.

## Features

- Generate pydantic models and examples compatible with GPT-4 structured output
- Endpoint to query with structured output
- Store models in `so_models` directory

## Usage

Visit `structured_output_4o.html` to access the web interface. You'll find:
- An input field for your query
- A generate button
- List of all generated pydantic models

## Module Export

The main module can be imported to:

1. Generate pydantic models
2. Store them in `so_models/` directory
3. (Optional) Generate and store examples

## Example
```python
from memos/_1_structured_output_4o.structured_output_4o import main as structured_output_4o_main

# Generate and save a model from query
model = structured_output_4o_main("your query here")
```