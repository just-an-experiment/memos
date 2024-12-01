from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
import os
import time
from enum import Enum
import ast

from memos._3_pure_function_so4o.pure_function_so4o import (
    Pure,
    FunctionSpec,
    GeneratedFunction,
    verify_pure_function,
    generate_pure_function as generate_pure_function_base
)


class ProgrammingLanguage(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"

    @property
    def file_extension(self) -> str:
        extensions = {
            self.PYTHON: ".py",
            self.JAVASCRIPT: ".js",
            self.TYPESCRIPT: ".ts",
            self.RUST: ".rs",
            self.GO: ".go",
        }
        return extensions.get(self, ".txt")

client = OpenAI()

# Global defaults
MAX_ATTEMPTS_DEFAULT = 3
NUM_TESTS_DEFAULT = 3
LANGUAGE_DEFAULT = ProgrammingLanguage.PYTHON

class InputParameter(BaseModel):
    """Represents a single input parameter with its value and type"""
    name: str = Field(..., description="Name of the parameter")
    value: str = Field(..., description="Value of the parameter")
    type: str = Field(..., description="Type of the parameter. It must be a built-in type, that doesn't require a module to be imported.")

class TestCase(BaseModel):
    """Represents a single test case with inputs and expected output"""
    inputs: list[InputParameter] = Field(
        ..., 
        description="List of input parameters with their values and types"
    )
    expected_output: str = Field(
        ..., 
        description="Expected output"
    )
    description: str = Field(
        ..., 
        description="Description of what this test case verifies",
    )


class GeneratedTests(BaseModel):
    """Represents generated tests for a function"""
    test_cases: list[TestCase] = Field(
        ..., 
        description="List of test cases",
    )
    test_code: str = Field(
        ..., 
        description="Generated test code implementing the test cases"
    )


def generate_tests(
    function_spec: FunctionSpec,
    language: ProgrammingLanguage,
    num_tests: int = NUM_TESTS_DEFAULT
) -> GeneratedTests:
    """Generate test cases for a pure function"""
    if num_tests < 1:
        raise ValueError("Number of tests must be positive")

    try:
        # Generate test cases using structured output
        test_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at generating comprehensive test cases for pure functions."},
                {"role": "user", "content": f"""Generate {num_tests} test cases for this function:

Function Spec:
- Name: {function_spec.function_name}
- Description: {function_spec.description}
- Input Parameters: {function_spec.input_params}
- Output Type: {function_spec.output_type}
- Language: {language.value}
Requirements:
1. Each test must include properly typed inputs and expected output
2. Include edge cases and normal cases
3. Provide clear descriptions of what each test verifies
4. Ensure expected output matches the function's return type
5. Do not import any modules
6. The test code should be ONLY a function named test_{function_spec.function_name}
7. The test code must call the provided function {function_spec.function_name} with the provided inputs and verify the output matches the expected output
8. Again, do not import any modules or write any other code than the test function
9. Don't redefine {function_spec.function_name} in the test code"""}
            ],
            response_format=GeneratedTests,
            timeout=30
        )
        generated_tests = test_completion.choices[0].message.parsed
        
        if not generated_tests.test_cases:
            raise ValueError("No test cases were generated")
        
        # Generate test code if not already provided
        if not generated_tests.test_code:
            generated_tests.test_code = generate_test_code(function_spec, generated_tests.test_cases, language)
        
        return GeneratedTests(
            test_cases=generated_tests.test_cases,
            test_code=generated_tests.test_code
        )
        
    except Exception as e:
        print(f"Error generating tests: {str(e)}")
        raise

def generate_test_code(
    function_spec: FunctionSpec,
    test_cases: list[TestCase],
    language: ProgrammingLanguage
) -> str:
    """Generate test function for the given function specification and test cases"""
    try:
        prompt = f"""Generate test function for this {language.value} function:

Function Specification:
- Name: {function_spec.function_name}
- Description: {function_spec.description}
- Input Parameters: {function_spec.input_params}
- Output Type: {function_spec.output_type}
- Language: {language.value}

Using these test cases:
{[test.model_dump() for test in test_cases]}

Requirements:
- The test function is named test_{function_spec.function_name}
- The test function must call {function_spec.function_name} with the provided inputs and verify the output matches the expected output
- Use {language.value}'s standard testing framework
- DO NOT import any modules
- Add descriptive comments
- Handle edge cases
- Don't redefine {function_spec.function_name} in the test code"""

        assistant = client.beta.assistants.create(
            instructions=prompt,
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}]
        )
        
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Generate the test function"
        )
        
        start_time = time.time()
        timeout = 30
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        while run.status not in ["completed", "failed"]:
            if time.time() - start_time > timeout:
                raise TimeoutError("Test code generation timed out")
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            time.sleep(1)
            
        if run.status == "failed":
            raise RuntimeError("Test code generation failed")
            
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_message = next(msg for msg in messages if msg.role == "assistant")
        
        if not assistant_message.content:
            raise ValueError("No test code was generated")
            
        return clean_generated_code(assistant_message.content[0].text.value, language)
        
    except Exception as e:
        print(f"Error generating test code: {str(e)}")
        raise
    
def clean_generated_code(code: str, language: ProgrammingLanguage) -> str:
    """Clean up generated code by removing markdown and extra whitespace"""
    code = code.strip()
    if code.startswith(f"```{language.value}"):
        code = code[len(f"```{language.value}"):].lstrip()
    elif code.startswith("```"):
        code = code[3:].lstrip()
    if code.endswith("```"):
        code = code[:-3].rstrip()
    return code

def save_function_and_tests(generated_function: GeneratedFunction, generated_tests: GeneratedTests, function_name: str, language: ProgrammingLanguage) -> None:
    """Save the generated function and tests to files"""
    try:
        base_dir = os.path.join('memos/_4_pure_function_with_tests_so4o', 'functions')
        os.makedirs(base_dir, exist_ok=True)
        
        safe_name = "".join(c for c in function_name if c.isalnum() or c in "_-")
        if not safe_name:
            raise ValueError("Invalid function name for file saving")
        function_path = os.path.join(base_dir, f"{safe_name}{language.file_extension}")
        
        with open(function_path, 'w') as f:
            f.write(generated_function.code)
            f.write("\n\n")
            f.write(generated_tests.test_code)
            f.write("\n\n")
            
            # Add language-specific exports
            if language == ProgrammingLanguage.PYTHON:
                f.write(f"export = {{\n    'tests': test_{safe_name},\n    'default': {safe_name}\n}}")
            elif language == ProgrammingLanguage.TYPESCRIPT:
                f.write(f"\n\nexport {{ test_{safe_name} as tests, {safe_name} as default }};")
            elif language == ProgrammingLanguage.JAVASCRIPT:
                f.write(f"\n\nmodule.exports = {{ tests: test_{safe_name}, default: {safe_name} }};")
            elif language == ProgrammingLanguage.RUST:
                f.write(f"\n\npub use test_{safe_name} as tests;\npub use {safe_name} as default;")
            elif language == ProgrammingLanguage.GO:
                # Go exports are handled through package-level declarations
                # Functions starting with uppercase are automatically exported
                pass
            
        print(f"Function and tests saved to {function_path}")
    except Exception as e:
        print(f"Error saving function and tests: {str(e)}")
        raise

def main(
    query: str,
    language: ProgrammingLanguage = LANGUAGE_DEFAULT,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT,
    num_tests: int = NUM_TESTS_DEFAULT
) -> tuple[GeneratedFunction, GeneratedTests]:
    """Main entry point for generating pure functions with tests"""
    try:
        # Generate the pure function
        generated_function = generate_pure_function_base(query, language, max_attempts)
        
        if generated_function.is_pure == Pure.NO:
            print("\nWarning: Generated function is not pure!")
        elif generated_function.is_pure == Pure.UNKNOWN:
            print("\nWarning: Purity of generated function could not be determined!")
            print("With this implementation, purity can only be determined for Python functions.")
            
        # Generate tests
        generated_tests = generate_tests(generated_function.spec, language, num_tests)
        
        # Save the results
        save_function_and_tests(generated_function, generated_tests, generated_function.spec.function_name, language)
        print("\nGenerated Pure Function:")
        print(generated_function.code)
        print("\nGenerated Tests:")
        print(generated_tests.test_code)
        
        return generated_function, generated_tests
        
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
        
    num_tests_input = input("\nEnter number of tests to generate (default: 3): ").strip()
    try:
        if not num_tests_input:
            print("Using default of 3 tests.")
            num_tests = 3
        else:
            num_tests = int(num_tests_input)
            print(f"Number of tests set to: {num_tests}")
    except ValueError:
        print(f"Invalid number '{num_tests_input}'. Using default: 3 tests.")
        num_tests = 3
        
    main(user_query, language, max_attempts, num_tests)

export = {
    "default": main,
    "generate_tests": generate_tests,
    "save_function_and_tests": save_function_and_tests,
    "ProgrammingLanguage": ProgrammingLanguage,
    "main": main
}