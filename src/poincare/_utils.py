from types import MethodType
from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar, overload

S = TypeVar("S")
P = ParamSpec("P")
R = TypeVar("R")


class class_and_instance_method(Generic[S, P, R]):
    def __init__(self, func: Callable[Concatenate[S, P], R]):
        self.func = func

    @overload
    def __get__(self, obj: None, cls: type[S]) -> Callable[P, R]:
        ...

    @overload
    def __get__(self, obj: S, cls: type[S]) -> Callable[P, R]:
        ...

    def __get__(self, obj, cls):  # type: ignore
        if obj is None:
            return MethodType(self.func, cls)
        else:
            return MethodType(self.func, obj)
