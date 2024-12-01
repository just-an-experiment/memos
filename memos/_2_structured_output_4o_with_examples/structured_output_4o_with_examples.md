# structured-output-4o-with-examples

Generates pydantic models for GPT-4 structured output with optional examples.

## Features

- Create GPT-4o structured output compatible pydantic models
- Generate example data that matches the model schema
- Web interface for model/example generation
- Store pydantic models in .py files in `so_models` directory

## Usage

Access web interface via `structured_output_4o_with_examples.html`:
- Enter query
- Set example count
- Generate model
- View saved models

## Example
```python
from memos/_2_structured_output_4o_with_examples.structured_output_4o_with_examples import main as structured_output_4o_with_examples_main

# Generate and save a model from query
model = structured_output_4o_with_examples_main("your query here", num_examples=5)
```
