"""
    GBMSimulator

An asset simulator where the prices follow a geometric brownian motion.
"""
struct GBMSimulator{T<:AbstractFloat} <: AssetSimulator
    S₀::Vector{T}       # Initial asset prices
    μ::Vector{T}        # Drifts
    σ::Vector{T}        # Volatilities
    ρ::Matrix{T}        # Correlations
    function GBMSimulator(S₀, μ, σ, ρ)
        @assert all(σ .> 0) "σ must be positive"
        @assert isposdef(ρ) "ρ is not a correlation matrix"
        return new{eltype(S₀)}(S₀, μ, σ, ρ)
    end
end

function simulate_economy(sim::GBMSimulator, T::Int=500; rng::Union{AbstractRNG,Nothing}=nothing)
    rng = isnothing(rng) ? Random.GLOBAL_RNG : rng
    @unpack S₀, μ, σ, ρ = sim
    dt = 1 # / T
    R = cholesky(ρ).L 
    ϵ = R * randn(rng, Float32, nassets(sim), T)
    ν = μ .- .5σ.^2
    r = hcat(
        zeros(eltype(ν), size(ν)), 
        cumsum(ν .* dt .+ diagm(σ) * ϵ .* √dt, dims=2)
    )
    # Return path of prices and log-returns
    S₀ .* exp.(r), r[:, 2:end]
end

function calibrate_gbm(df::DataFrame)
    # Create a matrix of N×T log-returns
    tickers = sort(unique(df.ticker))
    S₀ = Float32.(vcat([df.price[df.ticker .== t][1] for t ∈ tickers]...))
    r = Float32.(hcat([df.ret[df.ticker .== t] for t ∈ tickers]...))
    σ = Float32.(vec(std(r, dims=1)))
    μ = Float32.(vec(mean(r, dims=1)) + .5σ.^2)
    ρ = cor(r)
    GBMSimulator(S₀, μ, σ, ρ)
end