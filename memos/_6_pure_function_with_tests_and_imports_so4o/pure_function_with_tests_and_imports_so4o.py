from pydantic import BaseModel, Field
from openai import OpenAI
import os
import ast
from typing import List, Optional
from pathlib import Path
import time

from memos._4_pure_function_with_tests_so4o.pure_function_with_tests_so4o import (
    ProgrammingLanguage,
    GeneratedTests,
    generate_tests,
    TestCase
)

from memos._3_pure_function_so4o.pure_function_so4o import (
    Pure,
    FunctionSpec,
    GeneratedFunction,
    verify_pure_function,
    generate_pure_function as generate_pure_function_base
)

client = OpenAI()

def get_default_library_path() -> str:
    """Get the default library path - the functions directory in current module"""
    current_file = Path(__file__)
    functions_path = current_file.parent / "functions"  # Get the functions directory in current module
    functions_path.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
    return str(functions_path)

class ImportedFunction(BaseModel):
    """Represents a function that can be imported from the provided folders"""
    name: str = Field(..., description="Name of the function")
    module_path: str = Field(..., description="Path to the module containing the function")
    description: str = Field(..., description="Brief description of what the function does")
    signature: str = Field(..., description="Function signature including parameters and return type")

class FunctionTree(BaseModel):
    """Represents the tree of available functions in the provided folders"""
    functions: List[ImportedFunction] = Field(default_factory=list, description="List of available functions")
    
class RelevantImports(BaseModel):
    """Model for the list of relevant imports"""
    imports: list[str] = Field(
        ..., 
        description="List of import statements for relevant functions"
    )

def extract_function_info(file_path: Path) -> List[ImportedFunction]:
    """Extract function information from a Python file"""
    try:
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if not content.strip():
            print(f"Empty file: {file_path}")
            return []
            
        tree = ast.parse(content)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function signature
                params = [arg.arg for arg in node.args.args]
                returns = ""
                if node.returns:
                    returns = ast.unparse(node.returns)
                signature = f"{node.name}({', '.join(params)}) -> {returns}"
                
                # Extract docstring
                docstring = ast.get_docstring(node) or "No description available"
                
                functions.append(ImportedFunction(
                    name=node.name,
                    module_path=str(file_path),
                    description=docstring,
                    signature=signature
                ))
                
        return functions
    except Exception as e:
        print(f"Error extracting function info from {file_path}: {str(e)}")
        return []

def build_function_tree(folders: List[str]) -> FunctionTree:
    """Build a tree of available functions from the provided folders"""
    tree = FunctionTree()
    default_library_path = get_default_library_path()
    
    if not folders:
        print("Warning: No folders provided")
        return tree
        
    valid_folders = []
    for folder in folders:
        full_path = os.path.join(default_library_path, folder)
        if not os.path.exists(full_path):
            print(f"Warning: Folder {full_path} does not exist")
            continue
        if not os.path.isdir(full_path):
            print(f"Warning: Path {full_path} is not a directory")
            continue
        valid_folders.append(full_path)
    
    if not valid_folders:
        print("Warning: No valid folders found in the default library path")
        return tree
        
    for folder in valid_folders:
        try:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        try:
                            functions = extract_function_info(file_path)
                            tree.functions.extend(functions)
                        except Exception as e:
                            print(f"Error processing file {file_path}: {str(e)}")
                            continue
        except Exception as e:
            print(f"Error walking folder {folder}: {str(e)}")
            continue
                    
    return tree

def generate_imports(function_tree: FunctionTree, query: str) -> List[str]:
    """Generate relevant import statements based on the query and available functions"""
    try:
        imports_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at identifying relevant function imports based on a query."},
                {"role": "user", "content": f"""Given this query: "{query}"
                And these available functions:
                {[f.model_dump() for f in function_tree.functions]}
                
                Return a list of import statements for relevant functions.
                Only include functions that would be helpful for implementing the query.
                Use relative imports with proper module paths."""}
            ],
            response_format=RelevantImports,
        )
        
        if imports_completion.choices[0].message.refusal:
            print(f"Model refused to generate imports: {imports_completion.choices[0].message.refusal}")
            return []
            
        return imports_completion.choices[0].message.parsed.imports
        
    except Exception as e:
        print(f"Error generating imports: {str(e)}")
        return []

def generate_pure_function(
    query: str,
    language: ProgrammingLanguage,
    function_tree: FunctionTree,
    max_attempts: int = 3
) -> GeneratedFunction:
    """Generate a pure function with imports based on the query"""
    try:
        # Add timeout handling
        start_time = time.time()
        timeout = 60  # 60 second timeout
        
        # First get function specifications using structured output
        spec_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": f"You are an expert at designing pure functions in {language.value}. Extract specifications for a pure function based on the user's query."},
                {"role": "user", "content": query}
            ],
            response_format=FunctionSpec,
        )
        
        spec = spec_completion.choices[0].message.parsed
        
        # Generate relevant imports based on the function spec
        imports_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at identifying relevant function imports based on function specifications."},
                {"role": "user", "content": f"""Given this function specification:
                - Name: {spec.function_name}
                - Description: {spec.description}
                - Input params: {spec.input_params}
                - Output type: {spec.output_type}
                
                And these available functions:
                {[f.model_dump() for f in function_tree.functions]}
                
                Return a list of import statements for relevant functions.
                Only include functions that would be helpful for implementing this specific function.
                Use relative imports with proper module paths."""}
            ],
            response_format=RelevantImports,
        )
        
        imports = []
        if not imports_completion.choices[0].message.refusal:
            imports = imports_completion.choices[0].message.parsed.imports
        imports_code = "\n".join(imports) + "\n\n" if imports else ""
        
        # Generate the actual function using code interpreter, including imports in context
        assistant = client.beta.assistants.create(
            instructions=f"""You are an expert at writing pure functions. Generate a pure function based on these specifications:
            - Name: {spec.function_name}
            - Description: {spec.description}
            - Language: {language.value}
            - Input params: {spec.input_params}
            - Output type: {spec.output_type}
            
            The following functions are available to use (already imported):
            {imports_code if imports_code else "No imported functions available."}
            
            Rules:
            1. ONLY include the pure function (same output for same input, no side effects)
            2. Include type hints if the language supports them
            3. Include a docstring explaining the function
            5. DO NOT include any imports, prints, or example usage.
            6. In case it wasn't clear, ONLY include the pure function.

            """,
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}]
        )
        
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"""Generate the pure function including these imports if they are used:

{imports_code if imports_code else "No imports needed."}

The function should match the specifications above."""
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        while run.status != "completed":
            if time.time() - start_time > timeout:
                raise TimeoutError("Function generation timed out")
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            time.sleep(1)
        
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_message = next(msg for msg in messages if msg.role == "assistant")
        
        # Extract code and clean it
        code = assistant_message.content[0].text.value
        code = code.strip()
        if code.startswith(f"```{language.value}"):
            code = code[len(f"```{language.value}"):].lstrip()
        if code.endswith("```"):
            code = code[:-len("```")].rstrip()
        
        if imports_code:
            code = imports_code + "\n\n" + code
        
        # Verify function is pure
        is_pure = verify_pure_function(code, spec, language)
        
        return GeneratedFunction(
            spec=spec,
            code=code,
            is_pure=is_pure,
            language=language.value
        )
        
    except Exception as e:
        print(f"Error generating function: {str(e)}")
        return GeneratedFunction(
            spec=FunctionSpec(
                function_name="failed_generation",
                description=f"Failed to generate function: {str(e)}",
                input_params=[],
                output_type="None"
            ),
            language=language.value,
            code="# Generation failed",
            is_pure=Pure.NO
        )

def generate_pure_function_with_imports(
    query: str,
    folders: List[str],
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    max_attempts: int = 3
) -> GeneratedFunction:
    """Generate a pure function with relevant imports from provided folders"""
    
    # Build function tree from folders
    function_tree = build_function_tree(folders)
    
    # Generate the function with imports
    for attempt in range(max_attempts):
        generated_function = generate_pure_function(query, language, function_tree)
        if generated_function.is_pure != Pure.NO:
            return generated_function
        print(f"Attempt {attempt + 1} failed to generate a pure function. Retrying...")
    
    return generated_function

def main(
    query: str,
    folders: List[str],
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    max_attempts: int = 3,
    num_tests: int = 3
) -> tuple[GeneratedFunction, GeneratedTests]:
    """Main entry point for generating pure functions with tests and imports"""
    try:
        # Generate function with imports
        generated_function = generate_pure_function_with_imports(
            query=query,
            folders=folders,
            language=language,
            max_attempts=max_attempts
        )
        
        if generated_function.is_pure == Pure.NO:
            print("\nWarning: Generated function is not pure!")
        elif generated_function.is_pure == Pure.UNKNOWN:
            print("\nWarning: Purity of generated function could not be determined!")
            
        # Generate tests
        generated_tests = generate_tests(generated_function.spec, language, num_tests)
        
        # Save results in the functions directory
        save_dir = get_default_library_path()
        
        file_path = os.path.join(save_dir, f"{generated_function.spec.function_name}{language.file_extension}")
        with open(file_path, 'w') as f:
            f.write(generated_function.code)
            f.write("\n\n")
            f.write(generated_tests.test_code)
            
        print("\nGenerated Pure Function:")
        print(generated_function.code)
        print("\nGenerated Tests:")
        print(generated_tests.test_code)
        
        return generated_function, generated_tests
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    # Interactive CLI
    while True:
        user_query = input("Enter your query to generate a pure function: ").strip()
        if user_query:
            break
        print("Please specify a query.")
    
    default_path = get_default_library_path()
    print(f"\nEnter folders in library to search for functions (comma-separated, '.' for functions directory):")
    print(f"Default: {default_path}")
    print(f"Note: Functions will be imported from and saved to: {default_path}")
    folders_input = input().strip()
    
    if not folders_input or folders_input == '.':
        folders = [default_path]
    else:
        folders = [Path(f.strip()).resolve() for f in folders_input.split(',') if f.strip()]
        folders = [str(f) for f in folders]  # Convert Path objects back to strings

    print("\nAvailable languages:")
    for lang in ProgrammingLanguage:
        print(f"- {lang.value}")
    lang_input = input("\nEnter programming language (default: python): ").lower().strip()
    
    try:
        language = ProgrammingLanguage(lang_input) if lang_input else ProgrammingLanguage.PYTHON
        print(f"Selected language: {language.value}")
    except ValueError:
        print(f"Invalid language '{lang_input}'. Using default: Python.")
        language = ProgrammingLanguage.PYTHON
        
    max_attempts = int(input("\nEnter maximum attempts (default: 3): ").strip() or "3")
    num_tests = int(input("\nEnter number of tests to generate (default: 3): ").strip() or "3")
    
    main(user_query, folders, language, max_attempts, num_tests)

export = {
    "default": main,
    "generate_pure_function_with_imports": generate_pure_function_with_imports,
    "build_function_tree": build_function_tree,
    "ProgrammingLanguage": ProgrammingLanguage,
    "main": main
}
