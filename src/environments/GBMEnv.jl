mutable struct GBMEnv <: SimulatorEnv
    sim::GBMSimulator
    t::Int                              # Current timestep
    T::Int                              # Max. timestep
    w::Vector{Float32}                  # Current portfolio weights
    fee::Float32                        # Percentage transaction fee
    reward::Float32                     # Instantaneous reward
    S::Matrix{Float32}                  # Price paths
    r::Matrix{Float32}                  # Log-return paths
    actions::Vector{Vector{Float32}}    # Vector of taken actions
    function GBMEnv(sim, T, fee)
        w = uniform_weights(nassets(sim))
        S, r = simulate_economy(sim, T)
        new(sim, 1, T, w, fee, 0f0, S, r, Vector{Float32}[])
    end
end

RLBase.state(env::GBMEnv) = 1
# RLBase.state(env::GBMEnv) = (f=env.f[nassets(env)+1:end, env.t], w=env.w)

function RLBase.reset!(env::GBMEnv)
    env.t = 1
    env.S, env.r = simulate_economy(env.sim, env.T)
    env.w = init_weights(env)
    env.actions = Float32[]
    env.reward = 0f0
    nothing
end

function (env::GBMEnv)(action)
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
    # Compute rewards, update capital
    env.reward = log(r * (1 - tc))
    env.t += 1
    nothing
end