# Represents the entire chain of thought process for solving an elementary school math problem.

from pydantic import BaseModel

class Step(BaseModel):
    """
    Represents a single step in the chain of thought for solving a math problem.
    """
    description: str
    """A detailed explanation of what is done in this step."""
    
    action: str
    """The specific mathematical operation or logical reasoning applied during this step."""


class ChainOfThought(BaseModel):
    """
    Represents the entire chain of thought process for solving an elementary school math problem.
    """
    problem_statement: str
    """The statement of the math problem to be solved."""
    
    steps: list[Step]
    """A structured sequence of steps outlining the thought process in solving the problem."""
    
    final_solution: str
    """The final answer or solution obtained after completing all steps."""

export = ChainOfThought


