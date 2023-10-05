# Change Log

- Rename `times` to `save_at` in `Simulator.interact`.
- Fix typing in `Simulator.solve` to accept `ArrayLike`.
- Improve `__repr__` for `System` type to show the number of components.
- Change `Backend` from `Enum` to `Literal`. No need to import the enum from `.compile`.

## 0.2.0

- Allow different ODE solvers and implement wrappers for SciPy and numbalsoda.
- Rename `Time` to `Independent` and `times` to `save_at` in `Simulator.solve`.
  Allows custom names for the independent variable
  and adding more then one independent variable in the future.
- Allow Sequence of variables in `Simulator.transform`.
- Several improvements to `Simulator.interact`.
- Add `ipywidgets` to extra requirements.
- Several improvements in typing and error messages.
- Prevent subclassing concrete systems (for now).

## 0.1.0

- First release of Poincaré.
