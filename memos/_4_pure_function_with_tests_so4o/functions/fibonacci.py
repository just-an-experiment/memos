def fibonacci(n: int) -> int:
    """
    Calculates the nth Fibonacci number.

    :param n: An integer representing the position in the Fibonacci sequence.
    :return: The nth Fibonacci number.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n):
            a, b = b, a + b
        return b

def test_fibonacci():
    # Test case 1
    result = fibonacci(0)
    assert result == 0, f"Expected 0 but got {result} for n=0"
    
    # Test case 2
    result = fibonacci(1)
    assert result == 1, f"Expected 1 but got {result} for n=1"
    
    # Test case 3
    result = fibonacci(10)
    assert result == 55, f"Expected 55 but got {result} for n=10"
    
    print("All test cases passed!")

export = { 'tests': test_fibonacci, 'default': fibonacci }