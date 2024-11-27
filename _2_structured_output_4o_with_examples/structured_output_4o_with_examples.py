from pydantic import BaseModel, Field
from openai import OpenAI
import os
import time

client = OpenAI()

imports = """from pydantic import BaseModel, Field

"""

import os

current_folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))


class GeneratedModel(BaseModel):
  model_query: str = Field(..., description="User's original query to generate the Pydantic model")
  model_code: str = Field(..., description="Python code for the generated Pydantic model")
  model_name: str = Field(..., description="Name of the generated Pydantic model")
  model_description: str = Field(..., description="Description of the pydantic class")
  examples: list[dict] = Field(default_factory=list, description="List of example dictionaries that match the model schema")


class ModelName(BaseModel):
  model_name: str = Field(..., description="Name of the primary pydantic class")
  model_description: str = Field(..., description="1 sentence description of the pydantic class")
  
def generate_model(query: str, num_examples: int = 0) -> GeneratedModel:
    # Validate input parameters
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    if not isinstance(num_examples, int) or num_examples < 0:
        raise ValueError("num_examples must be a non-negative integer")

    try:
        assistant = client.beta.assistants.create(
            instructions=f"""You are an expert at creating Pydantic models for OpenAI structured output. Generate Pydantic models based on the user's query. You may define nested Pydantic models if needed, though there should be one main one.
            
Rules:
1. ONLY include pydantic model definitions
2. Add descriptions to each field! 
3. In case #1 wasn't clear, DO NOT include imports, examples, prints, etc.
4. Assume that ONLY these imports are already available to you: {imports}
5. Again, PLEASE DO NOT INCLUDE IMPORTS
""",
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create assistant: {str(e)}")

    try:
        thread = client.beta.threads.create()
        
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Generate a Pydantic model for this query: {query}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create thread or message: {str(e)}")

    try:
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Wait for the run to complete with timeout
        start_time = time.time()
        timeout = 30  # 30 second timeout
        while run.status != "completed":
            if time.time() - start_time > timeout:
                raise TimeoutError("Model generation timed out")
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run.status == "failed":
                raise RuntimeError("Assistant run failed")
            time.sleep(1)
    except Exception as e:
        raise RuntimeError(f"Failed during model generation: {str(e)}")

    try:
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_message = next(msg for msg in messages if msg.role == "assistant")
        if not assistant_message.content:
            raise ValueError("Assistant returned empty response")
        
        model_code = assistant_message.content[0].text.value
        if not model_code:
            raise ValueError("Generated model code is empty")
    except StopIteration:
        raise ValueError("No assistant message found")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve model code: {str(e)}")

    # Clean the model code
    model_code = model_code.strip()
    if model_code.startswith("```python"):
        model_code = model_code[len("```python"):].lstrip()
    if model_code.endswith("```"):
        model_code = model_code[:-len("```")].rstrip()
    
    if not model_code:
        raise ValueError("Cleaned model code is empty")

    # Execute the model code to get the Pydantic model class
    try:
        namespace = {}
        exec(imports + model_code, namespace)
    except Exception as e:
        raise ValueError(f"Failed to create Pydantic model - invalid Python code: {str(e)}")
    
    try:
        # Get the model name using structured output
        model_name_completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an expert at extracting Pydantic class names from python code. Get the variable name of the main class."},
                {"role": "user", "content": f"Code:\n\n{model_code}"}
            ],
            response_format=ModelName,
        )
        
        if model_name_completion.choices[0].message.refusal:
            raise ValueError(f"Model refused to identify model name: {model_name_completion.choices[0].message.refusal}")
        
        model_name = model_name_completion.choices[0].message.parsed.model_name
        model_description = model_name_completion.choices[0].message.parsed.model_description
        
        if not model_name or not model_description:
            raise ValueError("Failed to extract model name or description")
    except Exception as e:
        raise RuntimeError(f"Failed to extract model information: {str(e)}")

    # Verify the model exists and is a Pydantic model
    if model_name not in namespace:
        raise ValueError(f"Could not find model class {model_name} in generated code")
    ModelClass = namespace[model_name]
    if not issubclass(ModelClass, BaseModel):
        raise ValueError(f"{model_name} is not a Pydantic model")

    # Generate examples using structured output if requested
    examples = []
    if num_examples > 0:
        try:
            # Create an Examples model that wraps a list of the target model
            examples_model_code = f"""
class {model_name}Examples(BaseModel):
"""
            for i in range(num_examples):
                examples_model_code += f"""    item{i + 1}: {model_name} = Field(...,
        description=f"Example {i + 1} of {model_description}"
    )
"""

            # Execute the Examples model code
            exec(imports + model_code + examples_model_code, namespace)
            # Add all variables from the namespace to the current namespace
            for var_name, var_value in namespace.items():
                if not var_name.startswith('__'):  # Exclude built-in variables
                    globals()[var_name] = var_value
            
            # Generate examples using structured output
            examples_completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert at generating realistic example data."},
                    {"role": "user", "content": f"Generate {num_examples} realistic example(s) that match this model:\n\n{model_code}."}
                ],
                response_format=globals()[f"{model_name}Examples"],
            )
            
            if examples_completion.choices[0].message.refusal:
                raise ValueError(f"Model refused to generate examples: {examples_completion.choices[0].message.refusal}")
            
            examples = [getattr(examples_completion.choices[0].message.parsed, f"item{i + 1}").model_dump() for i in range(num_examples)]
            
            # Verify we got the requested number of examples
            if len(examples) != num_examples:
                raise ValueError(f"Generated {len(examples)} examples, but {num_examples} were requested")
        
        except Exception as e:
            print(f"Warning: Failed to generate examples: {str(e)}")
            examples = []  # Reset examples on failure
    
    # Create and return the GeneratedModel
    return GeneratedModel(
        model_query=query,
        model_name=model_name,
        model_description=model_description,
        model_code=model_code,
        examples=examples
    )

def save_model(generated_model: GeneratedModel) -> None:
    
    try:
        if not os.path.exists(f'{current_folder}/so_models'):
            os.makedirs(f'{current_folder}/so_models', exist_ok=True)
        
        filename = f"{current_folder}/so_models/{generated_model.model_name}.py"
        
        # Check if we can write to the directory
        if os.path.exists(filename):
            if not os.access(filename, os.W_OK):
                raise PermissionError(f"No write permission for {filename}")
        elif not os.access(os.path.dirname(filename), os.W_OK):
            raise PermissionError(f"No write permission for directory {os.path.dirname(filename)}")
        
        with open(filename, 'w') as f:
            f.write(f"# {generated_model.model_description}\n\n")
            f.write(imports)
            f.write(generated_model.model_code)
            if generated_model.examples:
                f.write("\n\n# Example data that matches the model schema\n")
                f.write("examples = [\n")
                for example in generated_model.examples:
                    f.write(f"    {repr(example)},\n")
                f.write("]\n")
            f.write(f"\n\nexport = {{")
            f.write(f"\n    'default': {generated_model.model_name},")
            f.write(f"\n    'examples': examples")
            f.write(f"\n}}")
        
        print(f"Model saved to {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {str(e)}")
  
    
def main(query: str, num_examples: int = 0) -> GeneratedModel:
    generated_model = generate_model(query, num_examples)
    save_model(generated_model)
    print("\nGenerated Model:")
    print(generated_model.model_code)
    if generated_model.examples:
        print("\nGenerated Examples:")
        print(generated_model.examples)
    return generated_model

if __name__ == "__main__":
    user_query = input("Enter your query to generate a Pydantic model: ")
    num_examples = int(input("How many examples would you like to generate? (0 for none): "))
    main(user_query, num_examples)

export = {
    "default": main,
    "generate_model": generate_model,
    "save_model": save_model,
    "main": main
}
