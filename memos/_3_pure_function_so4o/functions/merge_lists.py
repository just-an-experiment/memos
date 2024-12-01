# Merge two lists into one list, maintaining the order of elements from each list.

def merge_lists(list1: list, list2: list) -> list:
    """
    Merge two lists into one list, maintaining the order of elements from each list.

    Args:
    list1 (list): The first list to be merged.
    list2 (list): The second list to be merged.

    Returns:
    list: A new list containing all elements from list1 followed by all elements from list2.
    """
    return list1 + list2