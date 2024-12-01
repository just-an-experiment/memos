# Evaluates a mathematical expression represented as a string, supporting variables, basic arithmetic operations, and mathematical functions.

def evaluate_expression(expression: str, variables: dict[str, float]) -> float:
    """
    Evaluates a mathematical expression represented as a string, supporting variables,
    basic arithmetic operations, and mathematical functions.

    Args:
    expression (str): The mathematical expression to evaluate, which may include variables.
    variables (dict): A dictionary of variable names to their float values.

    Returns:
    float: The result of the evaluated expression.

    Raises:
    ValueError: If the expression contains invalid characters or operations.
    KeyError: If an expression contains variables that are not provided in the variables dictionary.
    """
    
    # Allowed names for eval - safe functions and variables
    safe_names = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'int': int,
        'float': float,
        **variables  # Include all variables in the safe namespace
    }
    
    # Evaluate the expression using eval with safe names
    try:
        result = eval(expression, {"__builtins__": None}, safe_names)
    except (NameError, SyntaxError, TypeError):
        raise ValueError("Invalid expression or unsupported operations.")
    
    return float(result)