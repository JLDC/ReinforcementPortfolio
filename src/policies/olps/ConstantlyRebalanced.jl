"""
    ConstantlyRebalanced <: AbstractPolicy

Constantly rebalanced portfolio. Reallocates to fixed weights at chosen frequency.
"""
@Base.kwdef mutable struct ConstantlyRebalanced <: AbstractPolicy
    w::Vector{Float64}
    rebalancing_freq::Int = 30
end

function (p::ConstantlyRebalanced)(env::SimulatorEnv)
    if is_firststep(env) || env.t % p.rebalancing_freq == 0
        p.w
    else
        env.w
    end
end

UniformCRP(env::SimulatorEnv, freq::Int=30) = ConstantlyRebalanced(uniform_weights(env), freq)