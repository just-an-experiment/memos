# html-so4o

Generate HTML from a user query using GPT-4 structured output.

## Features

- Generate semantic HTML and CSS based on natural language descriptions
- View generated HTML and CSS code
- Uses GPT-4o's structured output capabilities for reliable HTML generation

## Usage

Access web interface via `html-so4o.html`:
1. Enter your HTML request description
2. Click "Generate HTML"
3. View the live preview
4. Switch between preview and code views

## Example
```python
from html_so4o import main as html_so4o_main

# Generate HTML from query
result = html_so4o_main("Create a contact form with name, email, and message fields")
print(result.html)  # Generated HTML
print(result.css)   # Generated CSS
```

