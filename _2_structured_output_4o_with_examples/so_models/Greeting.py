# A model representing a greeting with its message, language, and sender information.

from pydantic import BaseModel, Field

class Greeting(BaseModel):
    message: str = Field(..., description="The greeting message.")
    language: str = Field(..., description="The language in which the greeting is expressed.")
    sender: str = Field(..., description="The name or identifier of the person sending the greeting.")

# Example data that matches the model schema
examples = [
    {'message': 'Hello, how are you today?', 'language': 'English', 'sender': 'Alice'},
    {'message': 'Bonjour, comment ça va?', 'language': 'French', 'sender': 'Jean'},
    {'message': 'Hola, ¿cómo estás?', 'language': 'Spanish', 'sender': 'Carlos'},
]


export = {
    'default': Greeting,
    'examples': examples
}