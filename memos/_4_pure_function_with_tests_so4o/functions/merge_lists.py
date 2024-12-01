def merge_lists(list1: list, list2: list) -> list:
    """
    Merges two lists into one by alternating elements from each list.
    If one list is longer, appends the remaining elements from that list.
    
    Parameters:
    - list1: First list of elements.
    - list2: Second list of elements.
    
    Returns:
    - A new list with elements from both lists alternated.
    """
    merged_list = []
    min_length = min(len(list1), len(list2))
    
    for i in range(min_length):
        merged_list.append(list1[i])
        merged_list.append(list2[i])
    
    merged_list.extend(list1[min_length:])
    merged_list.extend(list2[min_length:])
    
    return merged_list

def test_merge_lists():
    # Test case 1: Two lists of equal length
    list1 = [1, 3, 5]
    list2 = [2, 4, 6]
    assert merge_lists(list1, list2) == [1, 2, 3, 4, 5, 6], "Test Case 1 Failed"
    
    # Test case 2: First list longer than second
    list1 = [1, 3, 5, 7, 9]
    list2 = [2, 4, 6]
    assert merge_lists(list1, list2) == [1, 2, 3, 4, 5, 6, 7, 9], "Test Case 2 Failed"
    
    # Test case 3: First list is empty
    list1 = []
    list2 = [10, 20, 30]
    assert merge_lists(list1, list2) == [10, 20, 30], "Test Case 3 Failed"
    
    print("All test cases passed!")

export = { 'tests': test_merge_lists, 'default': merge_lists }