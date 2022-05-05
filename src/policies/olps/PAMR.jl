@Base.kwdef mutable struct PAMR{T<:AbstractFloat} <: AbstractPolicy
    w::Vector{T}
    ϵ::T
    rebalancing_freq::Int = 30
end

function (p::PAMR)(env::SimulatorEnv)
    if is_firststep(env) || env.t % p.rebalancing_freq == 0
        # Update portfolio weights according to PAMR strategy
        x̄ = mean_returns(env)
        τ = max(0, (p.w'x - p.ϵ) / sum(abs2, x - x̄))
        # Simplex projection constraint and return
        p.w = simplex_proj(p.w .- τ * (x .- x̄))
        p.w
    else
        env.w
    end
end

