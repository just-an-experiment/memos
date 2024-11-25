from pydantic import BaseModel, Field
from openai import OpenAI
import os
import time



client = OpenAI()

imports = """from pydantic import BaseModel

"""

class GeneratedModel(BaseModel):
  model_query: str = Field(..., description="User's original query to generate the Pydantic model")
  model_code: str = Field(..., description="Python code for the generated Pydantic model")
  model_name: str = Field(..., description="Name of the generated Pydantic model")
  model_description: str = Field(..., description="Description of the pydantic class")


class ModelName(BaseModel):
  model_name: str = Field(..., description="Name of the primary pydantic class")
  model_description: str = Field(..., description="1 sentence description of the pydantic class")
  
def generate_model(query: str) -> GeneratedModel:
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
    
    thread = client.beta.threads.create()
    
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Generate a Pydantic model for this query: {query}"
    )
    
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    
    # Wait for the run to complete
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(1)
    
    # Retrieve the assistant's response
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    assistant_message = next(msg for msg in messages if msg.role == "assistant")
    
    # Extract the model code from the assistant's response
    model_code = assistant_message.content[0].text.value
    
    # Strip "```python" from the beginning and "```" from the end of model_code
    model_code = model_code.strip()
    if model_code.startswith("```python"):
        model_code = model_code[len("```python"):].lstrip()
    if model_code.endswith("```"):
        model_code = model_code[:-len("```")].rstrip()
        
    # Get the model name using structured output
    model_name_completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert at extracting Pydantic class names from python code. Get the variable name of the main class."},
            {"role": "user", "content": f"Code:\n\n{model_code}"}
        ],
        response_format=ModelName,
    )
    # Check if the model refused to respond
    if model_name_completion.choices[0].message.refusal:
        print(f"Model refused to respond: {model_name_completion.choices[0].message.refusal}")
        model_name = None
        model_description = None
    else:
        model_name = model_name_completion.choices[0].message.parsed.model_name
        model_description = model_name_completion.choices[0].message.parsed.model_description
    
    # Create and return the GeneratedModel
    return GeneratedModel(
        model_query=query,
        model_name=model_name,
        model_description=model_description,
        model_code=model_code
    )

def save_model(generated_model: GeneratedModel) -> None:
    if not os.path.exists('structured_output_4o/so_models'):
        os.makedirs('structured_output_4o/so_models')
    
    filename = f"structured_output_4o/so_models/{generated_model.model_name}.py"
    with open(filename, 'w') as f:
        f.write(f"# {generated_model.model_description}\n\n")
        f.write(imports)
        f.write(generated_model.model_code)
        f.write(f"\n\nexport = {generated_model.model_name}")
    
    print(f"Model saved to {filename}")
  
    
def main(query: str, generate_example: bool = False) -> GeneratedModel:
    generated_model = generate_model(query)
    save_model(generated_model)
    print("\nGenerated Model:")
    print(generated_model.model_code)
    return generated_model

if __name__ == "__main__":
    user_query = input("Enter your query to generate a Pydantic model: ")
    main(user_query)

export = {
    "default": main,
    "generate_model": generate_model,
    "save_model": save_model,
    "main": main
}
