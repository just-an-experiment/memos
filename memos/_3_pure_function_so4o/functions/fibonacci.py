# Calculates the nth Fibonacci number using a recursive approach.

def fibonacci(n: int) -> int:
    """
    Calculates the nth Fibonacci number using a recursive approach.
    
    :param n: The position in the Fibonacci sequence.
    :return: The nth Fibonacci number.
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)