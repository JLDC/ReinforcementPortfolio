# Asset Simulators

The abstract supertype [`AssetSimulator`](@ref ReinforcementPortfolio.AssetSimulator) encompasses all simulators that can be used to generate the asset prices and underlying factors.

```@docs
ReinforcementPortfolio.AssetSimulator
```

Each simulator has its own asset returns dynamics, these are described in the docstring of the relevant simulator (see [Simulators](@ref Simulators)).

## Usage
In general, an [`AssetSimulator`](@ref ReinforcementPortfolio.AssetSimulator) is passed as an input to construct a RL environment of type [`SimulatorEnv`](@ref ReinforcementPortfolio.SimulatorEnv).

However, it is also possible to interact directly with the [`AssetSimulator`](@ref ReinforcementPortfolio.AssetSimulator) object. In particular, the `simulate_economy` function can be called with the simulator as input to generate a tuple of `(asset prices, asset returns, factors)`:

```@docs
simulate_economy
```


## Simulators
### GPSimulator

```@docs
GPSimulator
```

