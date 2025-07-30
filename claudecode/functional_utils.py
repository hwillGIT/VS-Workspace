"""
Functional Programming Utilities for ClaudeCode

This module provides functional programming patterns and utilities that can be used
across all projects in the ClaudeCode workspace. It emphasizes immutability,
pure functions, and functional composition.

Key principles:
- Pure functions (no side effects)
- Immutable data structures
- Function composition
- Lazy evaluation
- Higher-order functions
"""

import functools
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union,
    Iterator, Generator, Protocol
)
from itertools import chain, accumulate, islice, takewhile, dropwhile
from operator import add, mul, and_, or_
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import copy


T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class Functor(Protocol):
    """Protocol for functor-like operations."""
    def map(self, func: Callable[[T], U]) -> 'Functor[U]':
        ...


class Monad(Functor, Protocol):
    """Protocol for monad-like operations."""
    def flat_map(self, func: Callable[[T], 'Monad[U]']) -> 'Monad[U]':
        ...


@dataclass(frozen=True)
class Maybe:
    """
    Maybe monad for handling nullable values functionally.
    Eliminates null pointer exceptions and enables safe chaining.
    """
    value: Optional[T]
    
    @classmethod
    def some(cls, value: T) -> 'Maybe[T]':
        """Create a Maybe with a value."""
        if value is None:
            return cls.none()
        return cls(value)
    
    @classmethod
    def none(cls) -> 'Maybe[T]':
        """Create an empty Maybe."""
        return cls(None)
    
    def is_some(self) -> bool:
        """Check if Maybe contains a value."""
        return self.value is not None
    
    def is_none(self) -> bool:
        """Check if Maybe is empty."""
        return self.value is None
    
    def map(self, func: Callable[[T], U]) -> 'Maybe[U]':
        """Apply function if value exists, otherwise return None."""
        if self.is_none():
            return Maybe.none()
        try:
            return Maybe.some(func(self.value))
        except Exception:
            return Maybe.none()
    
    def flat_map(self, func: Callable[[T], 'Maybe[U]']) -> 'Maybe[U]':
        """Apply function that returns Maybe, flattening the result."""
        if self.is_none():
            return Maybe.none()
        try:
            return func(self.value)
        except Exception:
            return Maybe.none()
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Maybe[T]':
        """Keep value only if predicate is true."""
        if self.is_none() or not predicate(self.value):
            return Maybe.none()
        return self
    
    def get_or_else(self, default: T) -> T:
        """Get value or return default."""
        return self.value if self.is_some() else default
    
    def or_else(self, alternative: 'Maybe[T]') -> 'Maybe[T]':
        """Return this Maybe if it has value, otherwise return alternative."""
        return self if self.is_some() else alternative


@dataclass(frozen=True)
class Either:
    """
    Either monad for error handling without exceptions.
    Represents a value that can be either a success (Right) or failure (Left).
    """
    left: Optional[Any] = None
    right: Optional[T] = None
    
    @classmethod
    def left(cls, error: Any) -> 'Either[Any, T]':
        """Create a Left (error) value."""
        return cls(left=error)
    
    @classmethod
    def right(cls, value: T) -> 'Either[Any, T]':
        """Create a Right (success) value."""
        return cls(right=value)
    
    def is_left(self) -> bool:
        """Check if this is a Left (error) value."""
        return self.left is not None
    
    def is_right(self) -> bool:
        """Check if this is a Right (success) value."""
        return self.right is not None
    
    def map(self, func: Callable[[T], U]) -> 'Either[Any, U]':
        """Apply function to Right value, pass through Left unchanged."""
        if self.is_left():
            return Either.left(self.left)
        try:
            return Either.right(func(self.right))
        except Exception as e:
            return Either.left(e)
    
    def flat_map(self, func: Callable[[T], 'Either[Any, U]']) -> 'Either[Any, U]':
        """Apply function that returns Either, flattening the result."""
        if self.is_left():
            return Either.left(self.left)
        try:
            return func(self.right)
        except Exception as e:
            return Either.left(e)
    
    def get_or_else(self, default: T) -> T:
        """Get Right value or return default."""
        return self.right if self.is_right() else default
    
    def fold(self, left_func: Callable[[Any], U], right_func: Callable[[T], U]) -> U:
        """Apply appropriate function based on Left/Right state."""
        if self.is_left():
            return left_func(self.left)
        return right_func(self.right)


class FunctionalList:
    """
    Immutable list with functional operations.
    All operations return new instances without modifying the original.
    """
    
    def __init__(self, items: Iterable[T] = None):
        self._items = tuple(items) if items else ()
    
    def __iter__(self) -> Iterator[T]:
        return iter(self._items)
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __getitem__(self, index: int) -> T:
        return self._items[index]
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, FunctionalList):
            return False
        return self._items == other._items
    
    def __repr__(self) -> str:
        return f"FunctionalList({list(self._items)})"
    
    def map(self, func: Callable[[T], U]) -> 'FunctionalList[U]':
        """Apply function to all elements."""
        return FunctionalList(func(item) for item in self._items)
    
    def filter(self, predicate: Callable[[T], bool]) -> 'FunctionalList[T]':
        """Keep only elements that satisfy predicate."""
        return FunctionalList(item for item in self._items if predicate(item))
    
    def reduce(self, func: Callable[[U, T], U], initial: U = None) -> U:
        """Reduce list to single value using function."""
        if initial is None:
            return functools.reduce(func, self._items)
        return functools.reduce(func, self._items, initial)
    
    def fold_left(self, func: Callable[[U, T], U], initial: U) -> U:
        """Fold left with initial value."""
        return functools.reduce(func, self._items, initial)
    
    def fold_right(self, func: Callable[[T, U], U], initial: U) -> U:
        """Fold right with initial value."""
        return functools.reduce(lambda x, y: func(y, x), reversed(self._items), initial)
    
    def flat_map(self, func: Callable[[T], Iterable[U]]) -> 'FunctionalList[U]':
        """Map function and flatten results."""
        return FunctionalList(chain.from_iterable(func(item) for item in self._items))
    
    def flatten(self) -> 'FunctionalList':
        """Flatten nested iterables."""
        return FunctionalList(chain.from_iterable(self._items))
    
    def take(self, n: int) -> 'FunctionalList[T]':
        """Take first n elements."""
        return FunctionalList(islice(self._items, n))
    
    def drop(self, n: int) -> 'FunctionalList[T]':
        """Drop first n elements."""
        return FunctionalList(islice(self._items, n, None))
    
    def take_while(self, predicate: Callable[[T], bool]) -> 'FunctionalList[T]':
        """Take elements while predicate is true."""
        return FunctionalList(takewhile(predicate, self._items))
    
    def drop_while(self, predicate: Callable[[T], bool]) -> 'FunctionalList[T]':
        """Drop elements while predicate is true."""
        return FunctionalList(dropwhile(predicate, self._items))
    
    def group_by(self, key_func: Callable[[T], U]) -> Dict[U, 'FunctionalList[T]']:
        """Group elements by key function."""
        groups = {}
        for item in self._items:
            key = key_func(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        return {k: FunctionalList(v) for k, v in groups.items()}
    
    def partition(self, predicate: Callable[[T], bool]) -> Tuple['FunctionalList[T]', 'FunctionalList[T]']:
        """Partition into two lists based on predicate."""
        true_items = []
        false_items = []
        for item in self._items:
            if predicate(item):
                true_items.append(item)
            else:
                false_items.append(item)
        return FunctionalList(true_items), FunctionalList(false_items)
    
    def zip_with(self, other: 'FunctionalList[U]', func: Callable[[T, U], V]) -> 'FunctionalList[V]':
        """Zip with another list using function."""
        return FunctionalList(func(a, b) for a, b in zip(self._items, other._items))
    
    def append(self, item: T) -> 'FunctionalList[T]':
        """Append item to end."""
        return FunctionalList(list(self._items) + [item])
    
    def prepend(self, item: T) -> 'FunctionalList[T]':
        """Prepend item to beginning."""
        return FunctionalList([item] + list(self._items))
    
    def concat(self, other: 'FunctionalList[T]') -> 'FunctionalList[T]':
        """Concatenate with another list."""
        return FunctionalList(list(self._items) + list(other._items))
    
    def reverse(self) -> 'FunctionalList[T]':
        """Reverse the list."""
        return FunctionalList(reversed(self._items))
    
    def sort(self, key: Callable[[T], Any] = None, reverse: bool = False) -> 'FunctionalList[T]':
        """Sort the list."""
        return FunctionalList(sorted(self._items, key=key, reverse=reverse))
    
    def distinct(self) -> 'FunctionalList[T]':
        """Remove duplicates while preserving order."""
        seen = set()
        result = []
        for item in self._items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return FunctionalList(result)
    
    def head(self) -> Maybe[T]:
        """Get first element safely."""
        return Maybe.some(self._items[0]) if self._items else Maybe.none()
    
    def tail(self) -> 'FunctionalList[T]':
        """Get all elements except first."""
        return FunctionalList(self._items[1:]) if self._items else FunctionalList()
    
    def last(self) -> Maybe[T]:
        """Get last element safely."""
        return Maybe.some(self._items[-1]) if self._items else Maybe.none()
    
    def is_empty(self) -> bool:
        """Check if list is empty."""
        return len(self._items) == 0
    
    def find(self, predicate: Callable[[T], bool]) -> Maybe[T]:
        """Find first element matching predicate."""
        for item in self._items:
            if predicate(item):
                return Maybe.some(item)
        return Maybe.none()
    
    def exists(self, predicate: Callable[[T], bool]) -> bool:
        """Check if any element matches predicate."""
        return any(predicate(item) for item in self._items)
    
    def for_all(self, predicate: Callable[[T], bool]) -> bool:
        """Check if all elements match predicate."""
        return all(predicate(item) for item in self._items)


class FunctionalOps:
    """
    Functional programming operations and utilities.
    """
    
    @staticmethod
    def compose(*functions: Callable) -> Callable:
        """
        Compose functions right to left.
        compose(f, g, h)(x) = f(g(h(x)))
        """
        return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
    
    @staticmethod
    def pipe(*functions: Callable) -> Callable:
        """
        Pipe functions left to right.
        pipe(f, g, h)(x) = h(g(f(x)))
        """
        return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)
    
    @staticmethod
    def curry(func: Callable) -> Callable:
        """
        Transform function to be curried (partial application).
        """
        @functools.wraps(func)
        def curried(*args, **kwargs):
            if len(args) + len(kwargs) >= func.__code__.co_argcount:
                return func(*args, **kwargs)
            return lambda *more_args, **more_kwargs: curried(*(args + more_args), **{**kwargs, **more_kwargs})
        return curried
    
    @staticmethod
    def partial(func: Callable, *args, **kwargs) -> Callable:
        """
        Create partial function with some arguments pre-filled.
        """
        return functools.partial(func, *args, **kwargs)
    
    @staticmethod
    def memoize(func: Callable) -> Callable:
        """
        Memoize function results for performance.
        """
        return functools.lru_cache(maxsize=None)(func)
    
    @staticmethod
    def lazy_evaluate(func: Callable) -> Callable:
        """
        Create lazy evaluation wrapper.
        """
        result = None
        computed = False
        
        def evaluate():
            nonlocal result, computed
            if not computed:
                result = func()
                computed = True
            return result
        
        return evaluate
    
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
        """
        Decorator for retrying function calls.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            import time
                            time.sleep(delay)
                raise last_exception
            return wrapper
        return decorator
    
    @staticmethod
    def safe_call(func: Callable, *args, **kwargs) -> Either:
        """
        Call function safely, returning Either with result or error.
        """
        try:
            result = func(*args, **kwargs)
            return Either.right(result)
        except Exception as e:
            return Either.left(e)
    
    @staticmethod
    def chain_safe_calls(*funcs: Callable) -> Callable:
        """
        Chain function calls safely using Either monad.
        """
        def chained(initial_value):
            result = Either.right(initial_value)
            for func in funcs:
                result = result.flat_map(lambda x: FunctionalOps.safe_call(func, x))
            return result
        return chained


class ParallelOps:
    """
    Functional parallel processing operations.
    """
    
    @staticmethod
    def parallel_map(func: Callable[[T], U], items: Iterable[T], max_workers: int = None) -> List[U]:
        """
        Apply function to items in parallel.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, item) for item in items]
            return [future.result() for future in as_completed(futures)]
    
    @staticmethod
    async def async_map(func: Callable[[T], U], items: Iterable[T]) -> List[U]:
        """
        Apply async function to items concurrently.
        """
        if asyncio.iscoroutinefunction(func):
            tasks = [func(item) for item in items]
            return await asyncio.gather(*tasks)
        else:
            # Wrap sync function in async
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(None, func, item) for item in items]
            return await asyncio.gather(*tasks)
    
    @staticmethod
    def parallel_filter(predicate: Callable[[T], bool], items: Iterable[T], max_workers: int = None) -> List[T]:
        """
        Filter items in parallel.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [(item, executor.submit(predicate, item)) for item in items]
            return [item for item, future in futures if future.result()]


# Functional list factory function
def fl(items: Iterable[T] = None) -> FunctionalList[T]:
    """Create a FunctionalList from iterable."""
    return FunctionalList(items)


# Common functional operations
def fmap(func: Callable[[T], U], items: Iterable[T]) -> FunctionalList[U]:
    """Functional map operation."""
    return FunctionalList(items).map(func)


def ffilter(predicate: Callable[[T], bool], items: Iterable[T]) -> FunctionalList[T]:
    """Functional filter operation."""
    return FunctionalList(items).filter(predicate)


def freduce(func: Callable[[U, T], U], items: Iterable[T], initial: U = None) -> U:
    """Functional reduce operation."""
    return FunctionalList(items).reduce(func, initial)


def fpartition(predicate: Callable[[T], bool], items: Iterable[T]) -> Tuple[FunctionalList[T], FunctionalList[T]]:
    """Functional partition operation."""
    return FunctionalList(items).partition(predicate)


def fgroupby(key_func: Callable[[T], U], items: Iterable[T]) -> Dict[U, FunctionalList[T]]:
    """Functional group by operation."""
    return FunctionalList(items).group_by(key_func)


# Export all important classes and functions
__all__ = [
    'Maybe', 'Either', 'FunctionalList', 'FunctionalOps', 'ParallelOps',
    'fl', 'fmap', 'ffilter', 'freduce', 'fpartition', 'fgroupby'
]