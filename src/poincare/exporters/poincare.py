from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Sequence

from ..types import (
    Constant,
    Derivative,
    Equation,
    Node,
    Parameter,
    System,
    Variable,
)


@dataclass
class StringExporter:
    system: type[System] | None = None
    imports: defaultdict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    systems: dict[type[System], str] = field(default_factory=dict)

    @singledispatchmethod
    def convert(self, value) -> str:
        raise NotImplementedError(type(value))

    @convert.register
    def convert_constant(self, value: Constant) -> str:
        self.imports["poincare"].add("Constant")
        if value.name in self.system._annotations:
            self.imports["poincare"].add("assign")
            if value.default is None:
                rhs = "assign(constant=True)"
            else:
                rhs = f"assign(default={value.default}, constant=True)"
            return f"{value.name}: Constant = {rhs}"
        else:
            return f"{value.name} = Constant(default={value.default})"

    @convert.register
    def convert_parameter(self, value: Parameter) -> str:
        self.imports["poincare"].add("Parameter")
        if value.name in self.system._annotations:
            self.imports["poincare"].add("assign")
            if value.default is None:
                rhs = "assign()"
            else:
                rhs = f"assign(default={value.default})"
            return f"{value.name}: Parameter = {rhs}"
        else:
            return f"{value.name} = Parameter(default={value.default})"

    @convert.register
    def convert_variable(self, value: Variable) -> str:
        self.imports["poincare"].add("Variable")
        if value.name in self.system._annotations:
            self.imports["poincare"].add("initial")
            if value.initial is None:
                rhs = "initial()"
            else:
                rhs = f"initial(default={value.initial})"
            return f"{value.name}: Variable = {rhs}"
        else:
            return f"{value.name} = Variable(default={value.initial})"

    @convert.register
    def convert_derivative(self, value: Derivative) -> str:
        if value.initial is None:
            rhs = f"{value.variable.name}.derive()"
        else:
            rhs = f"{value.variable.name}.derive(initial={value.initial})"

        if value.name in self.system._annotations:
            self.imports["poincare"].add("Derivative")
            return f"{value.name}: Derivative = {rhs}"
        else:
            return f"{value.name} = {rhs}"

    @convert.register
    def convert_equation(self, eq: Equation) -> str:
        return f"{eq.name} = {eq.lhs} << {eq.rhs}"

    @convert.register
    def convert_system(self, system: System) -> str:
        if system.__class__ not in self.systems:
            self.convert_system_type(system.__class__)
        kwargs = ", ".join(f"{k}={v}" for k, v in system._kwargs.items())
        return f"{system.name} = {system.__class__.__name__}({kwargs})"

    def convert_system_type(self, system: type[System]):
        exporter = StringExporter(system, imports=self.imports, systems=self.systems)
        lines = [f"class {system.__name__}(System):"]
        for x in system._yield(Node, recursive=False):
            lines.append(exporter.convert(x))
        self.systems[system] = "\n    ".join(lines)
        return self


def to_string(system: type[System]) -> str:
    self = StringExporter()
    self.convert_system_type(system)
    imports_def = "\n".join(
        [
            to_import_string(name, sorted(values))
            for name, values in self.imports.items()
        ]
    )

    return "\n\n".join(
        [
            imports_def,
            *self.systems.values(),
        ]
    )


def to_import_string(name: str, value: Sequence[str] = ()) -> str:
    if len(value) == 0:
        return f"import {name}"
    else:
        values = ", ".join(value)
        return f"from {name} import {values}"


if __name__ == "__main__":
    from .. import assign, initial

    class SubSubModel1(System):
        x: Variable = initial()

    class SubSubModel2(System):
        y: Variable = initial()

    class SubModel(System):
        x: Variable = initial(default=0)
        s1 = SubSubModel1(x=0)
        s2 = SubSubModel2(y=0)

    class Model(System):
        x: Variable = initial(default=0)
        ss1 = SubSubModel1(x=0)
        vx: Derivative = x.derive(initial=0)
        y = Variable(initial=1)
        vy = y.derive(initial=1)
        z: Variable = initial()
        eq = vx.derive() << x
        eq2 = vy.derive() << x + y * 2
        s0 = SubModel()
        s1 = SubModel(x=0)
        s2 = SubModel(x=x)
        c: Constant = assign(default=0, constant=True)
        c1 = Constant(default=0)
        p: Parameter = assign(default=0)
        p1 = Parameter(default=0)

    print(to_string(Model))
