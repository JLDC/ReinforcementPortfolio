"""
    ConstantlyRebalanced(w, freq) <: AbstractPolicy

Constantly rebalanced portfolio. Reallocates to fixed weights `w` at chosen frequency `freq`.
"""
@Base.kwdef mutable struct ConstantlyRebalanced <: AbstractPolicy
    w::Vector{Float64}
    freq::Int = 30
end

function (p::ConstantlyRebalanced)(env::SimulatorEnv)
    if is_firststep(env) || env.t % p.freq == 0
        p.w
    else
        env.w
    end
end

"""
    UniformCRP(env, freq = 30)

Creates a uniform [`ConstantlyRebalanced`](@ref ReinforcementPortfolio.ConstantlyRebalanced)
strategy for the given environment `env` and rebalancing frequency `freq`.
"""
UniformCRP(env::SimulatorEnv, freq::Int=30) = ConstantlyRebalanced(uniform_weights(env), freq)