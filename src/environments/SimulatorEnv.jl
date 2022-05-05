abstract type SimulatorEnv <: AbstractEnv end

uniform_weights(env::SimulatorEnv) = uniform_weights(nassets(env))
init_weights(env::SimulatorEnv) = uniform_weights(env) # TODO: Initial weights -> uniform weights for now
logreturns(env::SimulatorEnv) = env.r[:, env.t]
mean_logreturns(env::SimulatorEnv) = mean(env.r[:, 1:env.t], dims=2) |> vec
returns(env::SimulatorEnv) = exp.(logreturns(env))
mean_returns(env::SimulatorEnv) = mean(exp.(env.r[:, 1:env.t]), dims=2) |> vec
is_firststep(env::SimulatorEnv) = env.t == 1


RLBase.is_terminated(env::SimulatorEnv) = env.t ≥ env.T
RLBase.action_space(env::SimulatorEnv) = SimplexSpace(nassets(env))
RLBase.state_space(env::SimulatorEnv) = StateSpace() # TODO
RLBase.reward(env::SimulatorEnv) = env.reward