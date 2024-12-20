from pydantic import BaseModel, Field
from openai import OpenAI
from enum import Enum
import os
import time

client = OpenAI()

class ModuleSpec(BaseModel):
    """Input specification for module generation"""
    description: str = Field(..., description="Description of what the module should do")
    module_context: list[str] = Field(default=None, description="Documents to help generate implementation")
    test_context: list[str] = Field(default=None, description="Documents to help generate test cases")
    name: str = Field(default=None, description="Optional module name")
    features: list[str] = Field(default=None, description="Optional list of features")
    dependencies: list[str] = Field(default=None, description="Optional list of dependencies")

class ModuleFile(BaseModel):
    """Represents a file in the generated module"""
    path: str = Field(..., description="Path to the file relative to project root")
    content: str = Field(..., description="Content of the file")

class TestCase(BaseModel):
    """Represents a test case for the module"""
    name: str = Field(..., description="Name of the test case")
    description: str = Field(..., description="Description of what the test verifies")
    code: str = Field(..., description="The actual test code")
    expected_result: str = Field(..., description="Expected outcome of the test")

class GeneratedModule(BaseModel):
    """Represents a complete generated memos module"""
    name: str = Field(..., description="Name of the module")
    description: str = Field(..., description="Description of what the module does")
    files: list[ModuleFile] = Field(..., description="List of files in the module")
    test_cases: list[TestCase] = Field(..., description="List of test cases")

class ModuleStep(str, Enum):
    """Steps in the module generation process"""
    DOCUMENTATION = "documentation"
    TEST_CASES = "test_cases"
    EXPORTS = "exports"
    MAIN_FUNCTION = "main_function"
    SUPPORTING_FUNCTIONS = "supporting_functions"
    HTML_TEMPLATE = "html_template"
    ROUTES = "routes"
    REFACTOR = "refactor"

class ModuleDocSpec(BaseModel):
    """Specification for module documentation"""
    title: str = Field(..., description="Title of the module")
    description: str = Field(..., description="Brief description of the module")
    features: list[str] = Field(..., description="List of key features")
    usage_example: str = Field(..., description="Python code example showing module usage")
    steps: list[str] = Field(..., description="List of steps the module takes")

def generate_module_documentation(description: str, module_context: list[str], module_name: str) -> ModuleFile:
    """Generates the module's markdown documentation"""
    try:
        # Generate documentation spec using structured output
        doc_spec_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at creating module documentation. Generate documentation specs based on the description and context."},
                {"role": "user", "content": f"Description: {description}\nContext:\n" + "\n".join(module_context or [])}
            ],
            response_format=ModuleDocSpec,
        )
        
        doc_spec = doc_spec_completion.choices[0].message.parsed
        
        # Generate the markdown content
        markdown_content = f"""# {doc_spec.title}

{doc_spec.description}

## Features

{chr(10).join(f'- {feature}' for feature in doc_spec.features)}

## Steps

{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(doc_spec.steps))}

## Usage

```python
{doc_spec.usage_example}
```

## Installation

After generating the module:

1. Add the imports to app.py:
```python
from memos.{module_name}.{module_name} import main as {module_name}_main
from memos.{module_name}.routes import routes as {module_name}_routes
```

2. Register the routes in app.py:
```python
# Add {module_name} routes
for route in {module_name}_routes:
    app.add_url_rule(
        route["rule"],
        route["endpoint"],
        route["view_func"],
        methods=route["methods"]
    )
```
"""
        
        # Create and return the ModuleFile
        return ModuleFile(
            path=f"memos/{module_name}/{module_name}.md",
            content=markdown_content
        )
        
    except Exception as e:
        print(f"Error generating documentation: {str(e)}")
        raise

class TestCaseSpec(BaseModel):
    """Specification for a test case"""
    name: str = Field(..., description="Name of the test")
    description: str = Field(..., description="What the test verifies")
    setup: str = Field(..., description="Setup code needed for the test")
    test_code: str = Field(..., description="The actual test code")
    expected_result: str = Field(..., description="Expected outcome")

def generate_test_cases(description: str, test_context: list[str]) -> list[TestCase]:
    """Generates test cases for the module"""
    try:
        # Generate test specifications using structured output
        test_specs_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": """You are an expert at creating comprehensive test cases. 
                Generate test specifications that cover:
                1. Basic functionality
                2. Edge cases
                3. Error handling
                4. Integration scenarios"""},
                {"role": "user", "content": f"Module Description: {description}\nTest Context:\n" + "\n".join(test_context or [])}
            ],
            response_format=list[TestCaseSpec],
        )
        
        test_specs = test_specs_completion.choices[0].message.parsed
        
        # Convert test specs to TestCase objects
        test_cases = []
        for spec in test_specs:
            full_test_code = f"""
# {spec.description}
import pytest
from typing import Dict, Any

@pytest.fixture
def {spec.name}_setup() -> Dict[str, Any]:
    {spec.setup}
    return locals()

def test_{spec.name}({spec.name}_setup: Dict[str, Any]):
    {spec.test_code}
"""
            test_cases.append(TestCase(
                name=spec.name,
                description=spec.description,
                code=full_test_code,
                expected_result=spec.expected_result
            ))
            
        return test_cases
        
    except Exception as e:
        print(f"Error generating test cases: {str(e)}")
        raise

class ExportSpec(BaseModel):
    """Specification for module exports"""
    main_function: str = Field(..., description="Name of the main function")
    supporting_functions: list[str] = Field(..., description="Names of supporting functions to export")
    classes: list[str] = Field(..., description="Names of classes to export")
    default: str = Field(..., description="Name of the default export")

class MainFunctionSpec(BaseModel):
    """Specification for the main function"""
    name: str = Field(..., description="Name of the main function")
    params: list[dict] = Field(..., description="List of parameters with name, type, and description")
    return_type: str = Field(..., description="Return type of the function")
    description: str = Field(..., description="Function description")
    implementation: str = Field(..., description="Implementation steps in Python code")

def generate_module_exports(
    main_function_name: str,
    supporting_functions: list[str],
    classes: list[str]
) -> str:
    """Generates the module's export interface based on actual implemented components"""
    try:
        exports_code = f"""
export = {{
    "default": {main_function_name},
    "main": {main_function_name},
    {','.join(f'"{func}": {func}' for func in supporting_functions)},
    {','.join(f'"{cls}": {cls}' for cls in classes)}
}}
"""
        return exports_code.strip()
        
    except Exception as e:
        print(f"Error generating exports: {str(e)}")
        raise

def generate_main_function(description: str) -> str:
    """Generates the module's main function"""
    try:
        # Generate main function specification using structured output
        main_spec_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": """You are an expert at designing main functions. 
                Generate a main function specification that:
                1. Has clear parameters with type hints
                2. Returns appropriate types
                3. Handles errors gracefully
                4. Follows Python best practices"""},
                {"role": "user", "content": f"Module Description: {description}"}
            ],
            response_format=MainFunctionSpec,
        )
        
        main_spec = main_spec_completion.choices[0].message.parsed
        
        # Generate the function code
        params_str = ", ".join(
            f"{p['name']}: {p['type']}" + (f" = {p.get('default', 'None')}" if p.get('default') else "")
            for p in main_spec.params
        )
        
        function_code = f"""
def {main_spec.name}({params_str}) -> {main_spec.return_type}:
    \"""{main_spec.description}\"""
    try:
{main_spec.implementation}
    except Exception as e:
        print(f"Error in {main_spec.name}: {{str(e)}}")
        raise
"""
        return function_code.strip()
        
    except Exception as e:
        print(f"Error generating main function: {str(e)}")
        raise

class SupportingFunctionSpec(BaseModel):
    """Specification for a supporting function"""
    name: str = Field(..., description="Name of the function")
    params: list[dict] = Field(..., description="List of parameters with name, type, and description")
    return_type: str = Field(..., description="Return type of the function")
    description: str = Field(..., description="Function description")
    implementation: str = Field(..., description="Implementation steps in Python code")

class HTMLTemplateSpec(BaseModel):
    """Specification for HTML template"""
    title: str = Field(..., description="Title for the webpage")
    description: str = Field(..., description="Description of the module")
    form_fields: list[dict] = Field(..., description="List of input fields")
    display_fields: list[dict] = Field(
        ..., 
        description="List of output fields with type (text, code, json)"
    )

def generate_supporting_functions(description: str, test_cases: list[TestCase]) -> list[str]:
    """Generates supporting functions needed by the module based on test cases"""
    try:
        # Generate supporting function specifications using structured output
        func_specs_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": """You are an expert at identifying and creating supporting functions.
                Analyze the module description and test cases to determine necessary helper functions."""},
                {"role": "user", "content": f"""
                Module Description: {description}
                Test Cases:
                {chr(10).join(f'- {test.name}: {test.description}' for test in test_cases)}
                """}
            ],
            response_format=list[SupportingFunctionSpec],
        )
        
        func_specs = func_specs_completion.choices[0].message.parsed
        
        # Generate function code for each specification
        functions = []
        for spec in func_specs:
            params_str = ", ".join(
                f"{p['name']}: {p['type']}" + (f" = {p.get('default', 'None')}" if p.get('default') else "")
                for p in spec.params
            )
            
            function_code = f"""
def {spec.name}({params_str}) -> {spec.return_type}:
    \"""{spec.description}\"""
    try:
{spec.implementation}
    except Exception as e:
        print(f"Error in {spec.name}: {{str(e)}}")
        raise
"""
            functions.append(function_code.strip())
            
        return functions
        
    except Exception as e:
        print(f"Error generating supporting functions: {str(e)}")
        raise

def generate_html_template(description: str, module_name: str) -> ModuleFile:
    """Generates the HTML template for the module"""
    try:
        # Generate HTML template specification using structured output
        template_spec_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at creating HTML templates for Flask applications."},
                {"role": "user", "content": f"Module Description: {description}"}
            ],
            response_format=HTMLTemplateSpec,
        )
        
        template_spec = template_spec_completion.choices[0].message.parsed
        
        # Generate the HTML template
        template_content = f"""
{{% extends "base.html" %}}

{{% block content %}}
<div class="container mt-5">
    <h1>{template_spec.title}</h1>
    <p class="lead">{template_spec.description}</p>

    <div class="card mb-4">
        <div class="card-body">
            <form id="generateForm" class="needs-validation" novalidate>
                {chr(10).join(f'''
                <div class="form-group mb-3">
                    <label for="{field['name']}">{field['label']}</label>
                    <input type="{field['type']}" 
                           class="form-control" 
                           id="{field['name']}" 
                           name="{field['name']}"
                           placeholder="{field.get('placeholder', '')}"
                           {field.get('required', 'required')}>
                </div>''' for field in template_spec.form_fields)}
                
                <button type="submit" class="btn btn-primary">Generate</button>
            </form>
        </div>
    </div>

    <div id="results" class="d-none">
        <div class="card">
            <div class="card-body">
                {chr(10).join(f'''
                <div class="mb-3">
                    <h3>{field['label']}</h3>
                    <div class="{field['class']}" id="{field['variable']}Container">
                        {{% if field['type'] == 'code' %}}
                        <pre><code id="{field['variable']}"></code></pre>
                        {{% else %}}
                        <div id="{field['variable']}"></div>
                        {{% endif %}}
                    </div>
                </div>''' for field in template_spec.display_fields)}
            </div>
        </div>
    </div>

    <div id="error" class="alert alert-danger d-none" role="alert">
    </div>
</div>

<script>
function updateResultField(field, value) {{
    const container = document.getElementById(`${{field.variable}}Container`);
    const element = document.getElementById(field.variable);
    
    switch (field.type) {{
        case 'code':
            element.textContent = value;
            hljs.highlightElement(element);
            break;
        case 'json':
            element.textContent = JSON.stringify(value, null, 2);
            break;
        default:
            element.textContent = value;
    }}
}}

document.getElementById('generateForm').addEventListener('submit', async (e) => {{
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    try {{
        const response = await fetch('/{module_name}/generate', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
            }},
            body: JSON.stringify(data)
        }});
        
        const result = await response.json();
        
        if (result.error) {{
            throw new Error(result.error);
        }}
        
        // Show results
        document.getElementById('results').classList.remove('d-none');
        document.getElementById('error').classList.add('d-none');
        
        // Update each result field
        {chr(10).join(f'''
        updateResultField({{"variable": "{field['variable']}", "type": "{field.get('type', 'text')}"}}, result.{field['variable']});''' 
        for field in template_spec.display_fields)}
        
    }} catch (error) {{
        document.getElementById('results').classList.add('d-none');
        const errorDiv = document.getElementById('error');
        errorDiv.textContent = error.message;
        errorDiv.classList.remove('d-none');
    }}
}});
</script>
{{% endblock %}}
"""
        
        return ModuleFile(
            path=f"templates/{module_name}.html",
            content=template_content.strip()
        )
        
    except Exception as e:
        print(f"Error generating HTML template: {str(e)}")
        raise

def generate_routes(module_name: str, main_function_name: str) -> ModuleFile:
    """Generates Flask routes for the module in a separate file"""
    try:
        route_code = f"""
from flask import render_template, request, jsonify
from . import {main_function_name}

def init_routes():
    routes = []
    
    def {module_name}_page():
        return render_template('{module_name}.html')
    routes.append({{
        "rule": '/{module_name}',
        "endpoint": "{module_name}_page",
        "view_func": {module_name}_page,
        "methods": ['GET']
    }})
    
    def {module_name}_generate():
        try:
            if not request.json:
                return jsonify({{"error": "Invalid request format"}}), 400
                
            result = {main_function_name}(**request.json)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({{"error": str(e)}}), 500
    routes.append({{
        "rule": '/{module_name}/generate',
        "endpoint": "{module_name}_generate",
        "view_func": {module_name}_generate,
        "methods": ['POST']
    }})
    
    return routes

routes = init_routes()
"""
        return ModuleFile(
            path=f"memos/{module_name}/routes.py",
            content=route_code.strip()
        )
        
    except Exception as e:
        print(f"Error generating routes: {str(e)}")
        raise

def save_module_files(generated_module: GeneratedModule) -> None:
    """Saves all generated module files to disk"""
    try:
        for file in generated_module.files:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file.path), exist_ok=True)
            
            # Save file
            with open(file.path, 'w') as f:
                f.write(file.content)
            print(f"Saved: {file.path}")
            
    except Exception as e:
        print(f"Error saving module files: {str(e)}")
        raise

def generate_module_name(name: str = None, description: str = None) -> str:
    """Generates a valid module name with prefix number"""
    try:
        # Get existing module numbers
        memos_dir = "memos"
        existing_modules = [d for d in os.listdir(memos_dir) 
                          if os.path.isdir(os.path.join(memos_dir, d)) 
                          and d.startswith('_')]
        
        # Get next module number
        next_num = 1
        if existing_modules:
            nums = [int(d.split('_')[1]) for d in existing_modules 
                   if len(d.split('_')) > 1 and d.split('_')[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            
        # Generate name from description if not provided
        if not name and description:
            name_completion = client.beta.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": """Generate a concise snake_case name for this module.
                    The name should:
                    1. Be descriptive of the main functionality
                    2. Use only lowercase letters, numbers, and underscores
                    3. Be between 2-4 words long
                    4. Not include generic terms like 'module' or 'generator'
                    Return only the name, nothing else."""},
                    {"role": "user", "content": description}
                ]
            )
            name = name_completion.choices[0].message.content.strip()
        
        if not name:
            name = "generated_module"
            
        # Clean name
        clean_name = "".join(c.lower() if c.isalnum() else '_' 
                           for c in name)
        clean_name = clean_name.strip('_')  # Remove leading/trailing underscores
        
        return f"_{next_num}_{clean_name}"
        
    except Exception as e:
        print(f"Error generating module name: {str(e)}")
        raise

def validate_dependencies(dependencies: list[str]) -> None:
    """Validate that all required dependencies are installed"""
    if not dependencies:
        return
        
    try:
        for dep in dependencies:
            if dep.startswith('import '):
                module = dep.split()[1].split('.')[0]
                __import__(module)
            elif dep.startswith('from '):
                module = dep.split()[1].split('.')[0]
                __import__(module)
    except ImportError as e:
        raise ImportError(f"Missing dependency: {str(e)}")

def validate_context(context: list[str], context_type: str) -> None:
    """Validate context documents format"""
    if not context:
        return
        
    for doc in context:
        if context_type == "module":
            if not any(line.strip().startswith(("def ", "class ")) for line in doc.split("\n")):
                raise ValueError(f"Module context must contain function or class definitions: {doc}")
        elif context_type == "test":
            if not any(line.strip().startswith("def test_") for line in doc.split("\n")):
                raise ValueError(f"Test context must contain test functions: {doc}")
            if not "assert" in doc:
                raise ValueError(f"Test context must contain assertions: {doc}")

def main(
    description: str | dict,
    module_context: list[str] = None,
    test_context: list[str] = None,
    name: str = None,
    features: list[str] = None,
    dependencies: list[str] = None
) -> GeneratedModule:
    """Main function to generate a complete memos module
    
    Args:
        description: Either a string description or a dict/ModuleSpec
        module_context: Optional list of example code snippets
        test_context: Optional list of example test cases
        name: Optional module name
        features: Optional list of features
        dependencies: Optional list of dependencies
    
    Returns:
        GeneratedModule: The generated module with all files and test cases
    """
    try:
        # Handle both direct parameters and dict/ModuleSpec input
        if isinstance(description, dict):
            spec = ModuleSpec(**description)
        else:
            spec = ModuleSpec(
                description=description,
                module_context=module_context,
                test_context=test_context,
                name=name,
                features=features,
                dependencies=dependencies
            )
            
        # Validate context documents
        validate_context(spec.module_context, "module")
        validate_context(spec.test_context, "test")
        
        # Validate dependencies first
        validate_dependencies(spec.dependencies)
        
        # Generate module name first
        module_name = generate_module_name(spec.name, spec.description)
        
        # Generate documentation
        doc_file = generate_module_documentation(spec.description, spec.module_context, module_name)
        
        # Generate test cases
        test_cases = generate_test_cases(spec.description, spec.test_context)
        
        # Generate main function
        main_function = generate_main_function(spec.description)
        main_function_name = main_function.split('def ')[1].split('(')[0].strip()
        
        # Generate supporting functions based on test cases
        supporting_functions = generate_supporting_functions(spec.description, test_cases)
        supporting_function_names = [f.split('def ')[1].split('(')[0].strip() 
                                   for f in supporting_functions]
        
        # Generate HTML template
        html_template = generate_html_template(spec.description, module_name)
        
        # Generate routes file
        routes_file = generate_routes(module_name, main_function_name)
        
        # Standard imports
        standard_imports = [
            "from pydantic import BaseModel, Field",
            "from flask import request, render_template, jsonify",
            "from typing import Dict, List, Optional, Any",
            "import os",
            "import json"
        ]
        
        # Add custom dependencies if provided
        if spec.dependencies:
            standard_imports.extend(spec.dependencies)
        
        # Combine all Python code
        python_code = "\n\n".join([
            "\n".join(standard_imports),
            "",  # Empty line after imports
            *supporting_functions,
            main_function,
            generate_module_exports(
                main_function_name,
                supporting_function_names,
                []  # No classes in this implementation
            )
        ])
        
        # Create Python module file
        python_file = ModuleFile(
            path=f"memos/{module_name}/{module_name}.py",
            content=python_code
        )
        
        # Create test file
        test_code = "\n\n".join([
            "import pytest",
            "from typing import Dict, Any",
            f"from .{module_name} import *",
            "",
            *[test.code for test in test_cases]
        ])
        
        test_file = ModuleFile(
            path=f"memos/{module_name}/test_{module_name}.py",
            content=test_code
        )
        
        # Create __init__.py
        init_file = ModuleFile(
            path=f"memos/{module_name}/__init__.py",
            content=f"from .{module_name} import *"
        )
        
        # Create complete module
        generated_module = GeneratedModule(
            name=module_name.lstrip('_'),  # Remove leading underscore for display
            description=spec.description,
            files=[
                doc_file,
                python_file,
                html_template,
                test_file,
                init_file,
                routes_file
            ],
            test_cases=test_cases
        )
        
        # Save all files
        save_module_files(generated_module)
        
        return generated_module
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

export = {
    "default": main,
    "generate_module_documentation": generate_module_documentation,
    "generate_test_cases": generate_test_cases,
    "generate_module_exports": generate_module_exports,
    "generate_main_function": generate_main_function,
    "generate_supporting_functions": generate_supporting_functions,
    "generate_html_template": generate_html_template,
    "generate_routes": generate_routes,
    "save_module_files": save_module_files,
    "main": main
}

if __name__ == "__main__":
    import sys
    from pprint import pprint
    
    # Example module spec
    example_spec = {
        "description": """Create a module that analyzes text and provides statistics like word count, 
        character count, most common words, and readability scores. The module should handle different 
        text formats, support multiple languages, and provide visualization of the results.""",
        
        "module_context": [
            """Example text analyzer:
            def analyze_text(text: str) -> Dict[str, Any]:
                return {
                    'word_count': len(text.split()),
                    'char_count': len(text)
                }
            """,
        ],
        
        "test_context": [
            """Test basic text analysis:
            def test_word_and_char_count():
                text = "Hello world. This is a test."
                result = analyze_text(text)
                assert result['word_count'] == 6
                assert result['char_count'] == 27
            """,
        ]
    }
    
    if len(sys.argv) > 1:
        # Get description from command line
        description = " ".join(sys.argv[1:])
        generated = main(description)
    else:
        # Use example spec
        print("\nGenerating example module...")
        generated = main(example_spec)
    
    print("\nGenerated Module Summary:")
    print(f"Module Name: {generated.name}")
    print(f"\nFiles Generated:")
    for file in generated.files:
        print(f"\n- {file.path}:")
        print("  " + "\n  ".join(file.content.split("\n")[:3]) + "\n  ...")
    
    print("\nTest Cases:")
    for test in generated.test_cases:
        print(f"\n- {test.name}: {test.description}")
        print("  First few lines of test code:")
        print("  " + "\n  ".join(test.code.split("\n")[:3]))
    
    print("\nTo use this module, add the following to app.py:")
    print(f"""
    from memos.{generated.name}.{generated.name} import main as {generated.name}_main
    from memos.{generated.name}.routes import routes as {generated.name}_routes
    
    # Add routes
    for route in {generated.name}_routes:
        app.add_url_rule(
            route["rule"],
            route["endpoint"],
            route["view_func"],
            methods=route["methods"]
        )
    """)
