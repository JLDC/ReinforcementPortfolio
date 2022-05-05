mutable struct GPEnv <: SimulatorEnv
    sim::Union{GPSimulator,GPSVSimulator}   # Simulator
    t::Int                                  # Current timestep
    T::Int                                  # Max. timestep
    w::Vector{Float32}                      # Current portfolio weights
    fee::Float32                            # Flat transaction fee
    reward::Float32                         # Instantaneous reward
    returns::Vector{Float32}                # Past log-returns
    S::Matrix{Float32}                      # Path of prices
    r::Matrix{Float32}                      # Path of log-returns
    f::Matrix{Float32}                      # Path of features
    actions::Vector{Vector{Float32}}        # Past actions (before simplex projection)
    reward_style::RewardStyle               # Reward function
    function GPEnv(sim, T, fee, reward_style)
        S, r, f = simulate_economy(sim, T)       # simulate_economy prices & features for the environment
        w = uniform_weights(nassets(sim))
        new(sim, 1, T, w, fee, 0f0, Float32[], S, r, f, Vector{Float32}[], reward_style)
    end
end

nfactors(env::GPEnv) = nfactors(env.sim)
nassets(env::GPEnv) = nassets(env.sim)

RLBase.state(env::GPEnv) = vcat(env.f[nassets(env)+1:end, env.t], env.w)
    
# (f=env.f[nassets(env)+1:end, env.t], w=env.w)

function RLBase.reset!(env::GPEnv)
    env.t = 1
    env.S, env.r, env.f = simulate_economy(env.sim, env.T)
    env.w = init_weights(env)
    env.actions = Float32[]
    env.reward = 0f0
    env.returns = Float32[]
    nothing
end

function (env::GPEnv)(action)
    # An action is taken at each CLOSE of the market. The state is directly updated with the 
    # new weights POST-MARKET MOVEMENTS, such that the state s' is the next day at CLOSE
    push!(env.actions, action)
    action′ = simplex_proj(action) # TODO: Make sure we want this, always project the action to the simplex
    # Compute transaction costs ------------------------------------------------------------
    tc = env.fee * sum(abs, env.w - action′)
    env.w = action′ # Reallocate portfolio (CLOSE)
    # Compute impact of market movements ---------------------------------------------------
    # Due to market movements throughout the next day, CLOSE weights change
    y = returns(env)
    r = env.w'y
    # Post market movements weights
    w′ = (env.w .* y) ./ r
    env.w = w′ # Update portfolio weights
    ret = log(r * (1 - tc))
    # Compute rewards
    if env.reward_style isa ReturnReward
        env.reward = ret
    elseif env.reward_style isa SharpeReward
        if env.t ≤ 10
            env.reward = 0f0 # TODO
        else
            env.reward = (sharpe_ratio(env.returns) - sharpe_ratio(vcat(env.returns, ret)))
        end
    end
    push!(env.returns, ret)
    env.t += 1
    nothing
end