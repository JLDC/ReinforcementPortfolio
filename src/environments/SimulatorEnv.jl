abstract type SimulatorEnv <: AbstractEnv end

nassets(env::SimulatorEnv) = nassets(env.sim)
uniform_weights(env::SimulatorEnv) = uniform_weights(nassets(env))
init_weights(env::SimulatorEnv) = uniform_weights(env) # Initial weights -> uniform weights for now
logreturns(env::SimulatorEnv) = env.r[:, env.t]
returns(env::SimulatorEnv) = exp.(logreturns(env))
is_firststep(env::SimulatorEnv) = env.t == 1


RLBase.is_terminated(env::SimulatorEnv) = env.t ≥ env.T
RLBase.action_space(env::SimulatorEnv) = SimplexSpace(nassets(env))
RLBase.state_space(env::SimulatorEnv) = StateSpace() # TODO
RLBase.reward(env::SimulatorEnv) = env.reward