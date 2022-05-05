abstract type PAMRPolicy <: AbstractPolicy end

@Base.kwdef mutable struct PAMR{T<:AbstractFloat} <: PAMRPolicy
    w::Vector{T}
    ϵ::T
    rebalancing_freq::Int = 30
end

@Base.kwdef mutable struct PAMR1{T<:AbstractFloat} <: PAMRPolicy
    w::Vector{T}
    ϵ::T
    C::Float32
    rebalancing_freq::Int = 30
end

@Base.kwdef mutable struct PAMR2{T<:AbstractFloat} <: PAMRPolicy
    w::Vector{T}
    ϵ::T
    C::Float32
    rebalancing_freq::Int = 30
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
    elseif env.t % p.rebalancing_freq == 0
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