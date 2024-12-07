from .functions.merge_lists import merge_lists

def merge_lists(lists: List[List[Any]]) -> List[Any]:
    """
    Merges two or more lists into a single list, preserving the order of elements.

    Parameters:
    lists (List[List[Any]]): A list of lists to be merged.
    
    Returns:
    List[Any]: A merged list containing all elements from the provided lists, in order.
    """
    merged_list = []
    for lst in lists:
        merged_list.extend(lst)
    return merged_list

def test_merge_lists():
    # Merging lists with integers, strings, and None
    result = merge_lists([[1, 2, 3], ['a', 'b', 'c'], [None]])
    assert result == [1, 2, 3, 'a', 'b', 'c', None], f"Unexpected result: {result}"