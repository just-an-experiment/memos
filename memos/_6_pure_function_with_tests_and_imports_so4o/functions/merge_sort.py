from .list_helpers.merge_lists import merge_lists



def merge_sort(list_of_elements: list[int]) -> list[int]:
    """
    Sorts a list of elements using the merge sort algorithm.
    
    Args:
        list_of_elements (list[int]): A list of integers to be sorted.
        
    Returns:
        list[int]: A new list containing the sorted elements.
    """
    if len(list_of_elements) <= 1:
        return list_of_elements
    
    mid = len(list_of_elements) // 2
    left_half = merge_sort(list_of_elements[:mid])
    right_half = merge_sort(list_of_elements[mid:])
    
    return merge_lists(left_half, right_half)

def test_merge_sort():
    # Test case 1
    result = merge_sort([5, 3, 8, 6, 2, 7, 4, 1])
    assert result == [1, 2, 3, 4, 5, 6, 7, 8], f"Test case 1 failed: {result}"
    
    # Test case 2
    result = merge_sort([])
    assert result == [], f"Test case 2 failed: {result}"
    
    # Test case 3
    result = merge_sort([10])
    assert result == [10], f"Test case 3 failed: {result}"
