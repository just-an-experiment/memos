from pydantic import BaseModel, Field
from openai import OpenAI
import os
import time
from enum import Enum
import ast
import inspect

client = OpenAI()

class ProgrammingLanguage(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    LEAN = "lean"

    @property
    def file_extension(self) -> str:
        extensions = {
            self.PYTHON: ".py",
            self.JAVASCRIPT: ".js",
            self.TYPESCRIPT: ".ts",
            self.RUST: ".rs",
            self.GO: ".go",
            self.LEAN: ".lean"
        }
        return extensions.get(self, ".txt")

# Global defaults
MAX_ATTEMPTS_DEFAULT = 3
LANGUAGE_DEFAULT = ProgrammingLanguage.PYTHON
    
class Pure(str, Enum):
  YES = "yes"
  NO = "no"
  UNKNOWN = "unknown"

class FunctionSpec(BaseModel):
    function_name: str = Field(..., description="Name of the pure function")
    description: str = Field(..., description="Brief description of what the function does")
    input_params: list[str] = Field(..., description="List of input parameter names and types")
    output_type: str = Field(..., description="Return type of the function")

class GeneratedFunction(BaseModel):
    spec: FunctionSpec = Field(..., description="Function specifications")
    language: ProgrammingLanguage = Field(..., description="Programming language of the generated function")
    code: str = Field(..., description="Generated function code")
    is_pure: Pure = Field(..., description="Whether the function is confirmed to be pure")

def generate_pure_function(
    query: str, 
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON, 
    max_attempts: int = MAX_ATTEMPTS_DEFAULT
) -> GeneratedFunction:
    try:
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
        
        # Generate the actual function using code interpreter
        assistant = client.beta.assistants.create(
            instructions=f"""You are an expert at writing pure functions. Generate a pure function based on these specifications:
            - Name: {spec.function_name}
            - Description: {spec.description}
            - Language: {language.value}
            - Input params: {spec.input_params}
            - Output type: {spec.output_type}
            
            Rules:
            1. ONLY include the pure function (same output for same input, no side effects)
            2. Include type hints if the language supports them
            3. Include a docstring explaining the function
            4. DO NOT include any imports or example usage
            5. In case #4 wasn't clear, DO NOT include imports, examples, prints, etc.
            """,
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}]
        )
        
        thread = client.beta.threads.create()
        
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user", 
            content="Generate the pure function based on the specifications above."
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        while run.status != "completed":
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
        
        # Verify function is pure
        is_pure = verify_pure_function(code, spec, language)
        
        attempts = 0
        while attempts < max_attempts:
            try:
                # Generate and verify function
                generated_function = GeneratedFunction(
                    spec=spec,
                    code=code,
                    is_pure=is_pure,
                    language=language.value
                )
                
                if generated_function.is_pure:
                    return generated_function
                    
                attempts += 1
                if attempts < max_attempts:
                    print(f"Attempt {attempts} failed to generate a pure function. Retrying...")
            
            except Exception as e:
                print(f"Error in attempt {attempts + 1}: {str(e)}")
                attempts += 1
                continue
        
        print(f"Failed to generate a pure function after {max_attempts} attempts.")
        return generated_function
        
    except Exception as e:
        print(f"Error generating function: {str(e)}")
        # Return a failed function generation
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

def verify_pure_function(code: str, spec: FunctionSpec, language: ProgrammingLanguage) -> Pure:
    """
    Verify if a function is pure by:
    1. Parsing the AST to check for side effects
    2. Running the function multiple times with same input to verify deterministic output
    """
    if language != ProgrammingLanguage.PYTHON:
        # For non-Python languages, assume pure since we can't easily verify
        return Pure.UNKNOWN
        
    try:
        # Parse the AST
        tree = ast.parse(code)
        
        # Check for common impure patterns
        class PurityChecker(ast.NodeVisitor):
            def __init__(self):
                self.is_pure = Pure.YES
                self.violations = []
                
            def visit_Call(self, node):
                # Check for calls to built-in impure functions
                if isinstance(node.func, ast.Name):
                    impure_functions = {'print', 'input', 'open', 'random', 'write'}
                    if node.func.id in impure_functions:
                        self.is_pure = Pure.NO
                        self.violations.append(f"Uses impure function: {node.func.id}")
                self.generic_visit(node)
                
            def visit_Attribute(self, node):
                # Check for file operations, system calls, etc.
                impure_attributes = {'write', 'append', 'pop', 'remove', 'clear'}
                if isinstance(node.attr, str) and node.attr in impure_attributes:
                    self.is_pure = Pure.NO
                    self.violations.append(f"Uses impure method: {node.attr}")
                self.generic_visit(node)
                
        checker = PurityChecker()
        checker.visit(tree)
        
        if not checker.is_pure:
            print("Function impurities found:", checker.violations)
            return Pure.NO
            
        # Execute the function to verify deterministic output
        namespace = {}
        exec(code, namespace)
        func_name = spec.function_name
        if func_name not in namespace:
            print("Could not find function in namespace")
            return Pure.NO
            
        func = namespace[func_name]
        
        # Try to execute with some basic inputs
        # This is a simple test - could be expanded based on input types
        test_inputs = [(1,), ("test",), ([1,2,3],)]
        for inputs in test_inputs:
            try:
                result1 = func(*inputs)
                result2 = func(*inputs)
                if result1 != result2:
                    print(f"Non-deterministic output detected for input {inputs}")
                    return Pure.NO
            except Exception as e:
                # Skip this test case if inputs aren't compatible
                continue
                
        return Pure.YES
        
    except Exception as e:
        print(f"Error verifying function purity: {str(e)}")
        return Pure.UNKNOWN

def save_function(generated_function: GeneratedFunction) -> None:
    try:
        if not os.path.exists('memos/_3_pure_function_so4o/functions'):
            os.makedirs('memos/_3_pure_function_so4o/functions')
        
        filename = f"memos/_3_pure_function_so4o/functions/{generated_function.spec.function_name}{generated_function.language.file_extension}"
        with open(filename, 'w') as f:
            f.write(generated_function.code)
        
        print(f"Function saved to {filename}")
    except Exception as e:
        print(f"Error saving function: {str(e)}")

def main(
    query: str, 
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT
) -> GeneratedFunction:
    try:
        generated_function = generate_pure_function(query, language, max_attempts)
        if generated_function.is_pure == Pure.YES:
            save_function(generated_function)
            print("\nGenerated Pure Function:")
            print(generated_function.code)
        elif generated_function.is_pure == Pure.NO:
            print("\nWarning: Generated function is not pure!")
        elif generated_function.is_pure == Pure.UNKNOWN:
            save_function(generated_function)
            print("\nWarning: Purity of generated function could not be determined!")
            print(" With this implementation, purity can only be determined for Python functions.")
        return generated_function
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    while True:
        user_query = input("Enter your query to generate a pure function: ").strip()
        if user_query:
            break
        print("Please specify a query.")
    print("\nAvailable languages:")
    for lang in ProgrammingLanguage:
        print(f"- {lang.value}")
    lang_input = input("\nEnter programming language (default: python): ").lower().strip()
    
    try:
        if not lang_input:
            print("Using Python as default.")
            language = ProgrammingLanguage.PYTHON
        else:
            language = ProgrammingLanguage(lang_input)
            print(f"Selected language: {language.value}")
    except ValueError:
        print(f"Invalid language '{lang_input}'. Using default: Python.")
        language = ProgrammingLanguage.PYTHON
        
    max_attempts_input = input("\nEnter maximum attempts (default: 3): ").strip()
    try:
        if not max_attempts_input:
            print("Using default of 3 attempts.")
            max_attempts = 3
        else:
            max_attempts = int(max_attempts_input)
            print(f"Maximum attempts set to: {max_attempts}")
    except ValueError:
        print(f"Invalid number '{max_attempts_input}'. Using default: 3 attempts.")
        max_attempts = 3
    main(user_query, language, max_attempts)

export = {
    "default": main,
    "generate_pure_function": generate_pure_function,
    "save_function": save_function,
    "ProgrammingLanguage": ProgrammingLanguage,
    "main": main
}
