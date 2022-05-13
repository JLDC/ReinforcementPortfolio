"""
    PAMRPolicy

Abstract type for Passive Aggressive Mean Reversion strategies.

Reference:
        B. Li, P. Zhao, S. C.H. Hoi, and V. Gopalkrishnan.
        Pamr: Passive aggressive mean reversion strategy for portfolio selection, 2012.
        https://link.springer.com/content/pdf/10.1007/s10994-012-5281-z.pdf
"""
abstract type PAMRPolicy <: AbstractPolicy end

"""
    PAMR(w, ϵ, freq)

Original Passive Aggressive Mean Reversion strategy. See 
[`PAMRPolicy`](@ref ReinforcementPortfolio.PAMRPolicy).
"""
@Base.kwdef mutable struct PAMR{T<:AbstractFloat} <: PAMRPolicy
    w::Vector{T}
    ϵ::T
    freq::Int = 30
end

"""
    PAMR1(w, ϵ, C, freq)

First variant of the Passive Aggressive Mean Reversion strategy. See 
[`PAMRPolicy`](@ref ReinforcementPortfolio.PAMRPolicy).
"""
@Base.kwdef mutable struct PAMR1{T<:AbstractFloat} <: PAMRPolicy
    w::Vector{T}
    ϵ::T
    C::T
    freq::Int = 30
end

"""
    PAMR2(w, ϵ, C, freq)

Second variant of the Passive Aggressive Mean Reversion strategy. See 
[`PAMRPolicy`](@ref ReinforcementPortfolio.PAMRPolicy).
"""
@Base.kwdef mutable struct PAMR2{T<:AbstractFloat} <: PAMRPolicy
    w::Vector{T}
    ϵ::T
    C::T
    freq::Int = 30
end

# Loss (same for all PAMR policies)
ℓₑ(p::PAMRPolicy, x::Vector{Float32}) = max(0, p.w'x - p.ϵ)
τₜ(p::PAMR, x::Vector{Float32}, x̄::Float32) = ℓₑ(p, x) / sum(abs2, x .- x̄)
τₜ(p::PAMR1, x::Vector{Float32}, x̄::Float32) = min(p.C, ℓₑ(p, x) / sum(abs2, x .- x̄))
τₜ(p::PAMR2, x::Vector{Float32}, x̄::Float32) = ℓₑ(p, x) / (sum(abs2, x .- x̄) + .5f0p.C)

function (p::PAMRPolicy)(env::SimulatorEnv)
    if is_firststep(env) # First step no relative price vector
        p.w = uniform_weights(env)
        p.w
    elseif env.t % p.freq == 0
        # Update portfolio weights according to PAMR strategy
        x̄ = mean(returns(env, env.t - 1))
        x = returns(env, env.t - 1)
        τ = τₜ(p, x, x̄)
        # Simplex projection constraint
        p.w = simplex_proj(p.w .- τ * (x .- x̄))
        p.w
    else
        env.w
    end
end