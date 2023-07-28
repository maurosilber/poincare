import math

import libsbml
from poincare import Parameter, Variable
from poincare.types import System
from simbio import Reaction, Species
from symbolite import Symbol

from .sbml_math import replace


class DynamicSystem:
    def __init__(self, name: str):
        self.system = type(name, (System,), {})

    def add(self, name: str, value):
        value.__set_name__(self.system, name)
        setattr(self.system, name, value)

    def __getattr__(self, name):
        return getattr(self.system, name)


def read_model(file: str, *, name: str | None = None):
    with open(file) as f:
        text = f.read()
    return parse_model(text, name=name)


def parse_model(sbml: str, *, name: str | None = None):
    document: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    if document.getNumErrors() != 0:
        raise RuntimeError("error reading the SBML file")

    model: libsbml.Model = document.getModel()
    if name is None:
        name: str = model.getName()
    Model = DynamicSystem(name)

    for p in model.getListOfParameters():
        add_parameter(Model, p)

    for s in model.getListOfSpecies():
        add_species(model, Model, s)

    for r in model.getListOfReactions():
        add_reaction(Model, r)

    return Model.system


def add_parameter(model: DynamicSystem, p: libsbml.Parameter):
    model.add(p.getId(), Parameter(default=p.getValue()))


def add_species(model: libsbml.Model, Model: DynamicSystem, s: libsbml.Species):
    if not math.isnan(initial := s.getInitialConcentration()):
        pass
    elif not math.isnan(initial := s.getInitialAmount()):
        pass
    else:
        assign: libsbml.InitialAssignment = model.getInitialAssignment(s.getId())
        math_ast: libsbml.ASTNode = assign.getMath()
        name = libsbml.formulaToL3String(math_ast)
        initial = getattr(Model, name)  # TODO: this must be a Constant, not Parameter

    Model.add(s.getId(), Variable(initial=initial))


def get_species(model: DynamicSystem, s: libsbml.SpeciesReference) -> Species:
    name = s.getSpecies()
    variable: Variable = getattr(model, name)
    st = s.getStoichiometry()
    if math.isnan(st):
        st = 1
    return Species(variable, st)


def formula_to_symbolite(Model: DynamicSystem, ast_node: libsbml.ASTNode):
    symbolite_node: Symbol = replace(ast_node)
    names = symbolite_node.symbol_names()
    mapper = {n: getattr(Model, n) for n in names}
    formula = symbolite_node.subs_by_name(**mapper)
    return formula


def add_reaction(Model: DynamicSystem, r: libsbml.Reaction):
    reactants = [get_species(Model, s) for s in r.getListOfReactants()]
    products = [get_species(Model, s) for s in r.getListOfProducts()]
    kinetic_law: libsbml.KineticLaw = r.getKineticLaw()
    formula: libsbml.ASTNode = kinetic_law.getMath()
    name = r.getName()
    if r.getReversible():
        assert formula.getType() == libsbml.AST_MINUS
        forward: libsbml.ASTNode = formula.getLeftChild()
        reverse: libsbml.ASTNode = formula.getRightChild()
        Model.add(
            f"{name}_forward",
            Reaction(
                reactants=reactants,
                products=products,
                rate_law=formula_to_symbolite(Model, forward),
            ),
        )
        Model.add(
            f"{name}_reverse",
            Reaction(
                reactants=products,
                products=reactants,
                rate_law=formula_to_symbolite(Model, reverse),
            ),
        )
    else:
        Model.add(
            name,
            Reaction(
                reactants=reactants,
                products=products,
                rate_law=formula_to_symbolite(Model, formula),
            ),
        )
