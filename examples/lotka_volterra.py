from poincare import Constant, System, Variable, assign, initial


class LoktaVolterra(System):
    prey: Variable = initial(default=0)
    predator: Variable = initial(default=0)

    prey_birth_rate: Constant = assign(default=0)
    prey_death_rate: Constant = assign(default=0)
    predator_death_rate: Constant = assign(default=0)
    predator_birth_rate: Constant = assign(default=0)

    birth_prey = prey.derive() << prey_birth_rate * prey
    death_prey = prey.derive() << -prey_death_rate * prey * predator

    birth_predator = predator.derive() << predator_birth_rate * prey * predator
    death_predator = predator.derive() << -predator_death_rate * predator
