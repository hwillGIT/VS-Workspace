"""
Tests for the LinkedList implementation.
"""

import pytest
from dspy.structures.linked_list import LinkedList


def test_empty_list():
    """Test creating an empty list."""
    ll = LinkedList()
    assert ll.is_empty()
    assert ll.size() == 0
    assert str(ll) == "Empty list"


def test_append():
    """Test appending elements to the list."""
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    
    assert not ll.is_empty()
    assert ll.size() == 3
    assert str(ll) == "1 -> 2 -> 3"


def test_prepend():
    """Test prepending elements to the list."""
    ll = LinkedList()
    ll.prepend(1)
    ll.prepend(2)
    ll.prepend(3)
    
    assert not ll.is_empty()
    assert ll.size() == 3
    assert str(ll) == "3 -> 2 -> 1"


def test_mixed_operations():
    """Test mixing append and prepend operations."""
    ll = LinkedList()
    ll.append(1)    # 1
    ll.prepend(2)   # 2 -> 1
    ll.append(3)    # 2 -> 1 -> 3
    ll.prepend(4)   # 4 -> 2 -> 1 -> 3
    
    assert ll.size() == 4
    assert str(ll) == "4 -> 2 -> 1 -> 3" 