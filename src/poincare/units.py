from typing import TypeVar

from pint.compat import fully_qualified_name, upcast_type_map

T = TypeVar("T", bound=type)


def register_with_pint(cls: T) -> T:
    """Register type with Pint to  to return NotImplemented
    on methods that can be reflected, such as __add__,
    by adding the type to pint's upcast map.
    """
    upcast_type_map[fully_qualified_name(cls)] = cls
    return cls
