"""
    SimulatorEnv <: AbstractEnv

Abstract supertype for reinforcement learning environments which are based on an
[`AssetSimulator`](@ref ReinforcementPortfolio.AssetSimulator)
"""
abstract type SimulatorEnv <: AbstractEnv end

uniform_weights(env::SimulatorEnv) = uniform_weights(nassets(env))
init_weights(env::SimulatorEnv) = uniform_weights(env) # TODO: Initial weights -> uniform weights for now
logreturns(env::SimulatorEnv) = env.r[:, env.t]
logreturns(env::SimulatorEnv, t::Int) = env.r[:, t]
returns(env::SimulatorEnv) = exp.(logreturns(env))
returns(env::SimulatorEnv, t::Int) = exp.(logreturns(env, t))
is_firststep(env::SimulatorEnv) = env.t == 1


RLBase.is_terminated(env::SimulatorEnv) = env.t ≥ env.T
RLBase.action_space(env::SimulatorEnv) = SimplexSpace(nassets(env))
RLBase.state_space(env::SimulatorEnv) = StateSpace() # TODO
RLBase.reward(env::SimulatorEnv) = env.reward