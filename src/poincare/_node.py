from __future__ import annotations

from typing import Iterator, TypeVar

from typing_extensions import Self

from ._utils import class_and_instance_method

T = TypeVar("T")


class Node:
    """Node objects are descriptors with a name and a parent.

    Setting an instance attribute (__set__),
    sets that instance as the parent of that attribute,
    if it did not have a previous parent set.
    Hence, it must be an Node subclass.

    Attribute access from an instance
    tries to return the object in the instance's __dict__.
    Otherwise, it creates a copy with self._copy_from,
    assigns the instance as its parent
    and saves is in the instance's __dict__.
    Hence, Node subclasses must implement self._copy_from.
    """

    name: str = ""
    parent: Node | None = None

    def _copy_from(self, parent: Node) -> Self:
        raise NotImplementedError

    def __set_name__(self, cls: Node, name: str):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parent", cls)

    def __set__(self, obj: Node, value: Self):
        if not isinstance(value, self.__class__):
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

        if value.parent is None and value.name == "":
            # if it has no name, it was created outside an EagerNamer
            value.__set_name__(obj, self.name)

        obj.__dict__[self.name] = value

    def __get__(self, parent: Node | None, cls: type[Node]) -> Self:
        if parent is None:
            return self

        try:
            return parent.__dict__[self.name]
        except KeyError:
            copy = self._copy_from(parent)
            copy.__set_name__(parent, self.name)
            parent.__dict__[self.name] = copy
            return copy

    def __str__(self) -> str:
        # This is a recursive method, as self.parent is Owned | None
        return f"{self.parent}.{self.name}"

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return str(self) == str(other)

    @class_and_instance_method
    def _yield(
        self,
        type: type[T],
        /,
        *,
        exclude: type | None = None,
        recursive: bool = True,
    ) -> Iterator[T]:
        if isinstance(self, Node):
            cls = self.__class__
        else:
            cls = self

        if exclude is None:
            exclude = ()  # type: ignore

        for k, v in cls.__dict__.items():
            if isinstance(v, type) and not isinstance(v, exclude):
                yield getattr(self, k)

            if recursive and isinstance(v, Node):
                v: Node = getattr(self, k)
                yield from v._yield(type, exclude=exclude, recursive=recursive)


class NodeMapper:
    def __init__(self, obj: Node):
        self.obj = obj
        self.cls = obj.__class__

    def get(self, item: T, default: T | None = None) -> T:
        if isinstance(item, Node) and item.parent is self.cls:
            return item.__get__(self.obj, self.cls)
        else:
            return item