"""
    GPSimulator(S₀, μ_f₀, σ_f₀, B, Σ, Ψ, Φ)

An asset simulator where the prices follow a model similar to that of [Gârleanu & Pedersen 
(2013)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12080)

The asset log returns adopt the following multidimensional factor model with ``N`` assets 
and ``K`` factors:

```math
\\begin{aligned}
\\mathbf{r}_{t+1} &= \\mathbf{B}\\mathbf{f}_t + \\mathbf{u}_{t+1} \\\\
\\mathbf{f}_{t+1} &= \\left(\\mathbf{I} - \\boldsymbol{\\Phi}\\right) \\mathbf{f}_t + \\boldsymbol{\\epsilon}_{t+1}.
\\end{aligned}
```

with

- ``\\mathbf{B} \\in \\mathbb{R}^{N \\times K}`` the matrix of factor loadings.
- ``\\boldsymbol{\\Phi} \\in \\mathbb{R}^{K \\times K}`` the matrix of mean-reversion coefficients 
- ``\\mathbf{u}_t \\sim \\mathcal{N}(\\mathbf{0}, \\boldsymbol{\\Sigma})`` the returns innovation term
- ``\\boldsymbol{\\epsilon}_t \\sim \\mathcal{N}(\\mathbf{0}, \\boldsymbol{\\Psi})`` the factors innovation term

When an economy is simulated according to the `GPSimulator`, the initial factors 
``\\mathbf{f}_0`` are randomized according to a normal distribution with mean `μ_f₀` and
standard deviation `σ_f₀`. It is assumed that the initial factor distribution is made of
independent normal distributions, i.e. `σ_f₀` is a vector of standard deviations and not
the covariance matrix (it is the diagonal of the covariance matrix).
"""
mutable struct GPSimulator{T<:AbstractFloat} <: AssetSimulator
    S₀::AbstractVector{T}       # Initial asset prices
    μ_f₀::AbstractVector{T}     # Initial factors means
    σ_f₀::AbstractVector{T}     # Initial factors standard dev.
    B::AbstractMatrix{T}        # Factor loadings
    Σ::AbstractMatrix{T}        # Covariance of u (noise term in returns)
    Ψ::AbstractMatrix{T}        # Covariance of ϵ (noise term in factors)
    Φ::AbstractMatrix{T}        # Mean-reversion coefficients
    function GPSimulator(S₀, μ_f₀, σ_f₀, B, Σ, Ψ, Φ)
        @assert ispossemidef(Σ) "Σ is not a valid covariance matrix"
        @assert ispossemidef(Ψ) "Ψ is not a valid covariance matrix"
        new{eltype(μ_f₀)}(S₀, μ_f₀, σ_f₀, B, Σ, Ψ, Φ)
    end
end

nfactors(sim::GPSimulator) = length(sim.μ_f₀)

"""
    simulate_economy(sim, T = 500; rng = nothing)

Simulates `T` periods of an economy according to the dynamics of the simulator `sim`.
"""
function simulate_economy(
    sim::GPSimulator, T::Int=500; 
    rng::Union{AbstractRNG,Nothing}=nothing, riskfree_asset::Bool = false
)
    rng = isnothing(rng) ? Random.GLOBAL_RNG : rng
    @assert T > 1 "T must be larger than 1"
    @unpack S₀, μ_f₀, σ_f₀, B, Σ, Ψ, Φ = sim
    N, K = size(B) # Extract number of assets / factors
    # Generate noise
    ϵ = vcat(
        zeros(eltype(μ_f₀), N, T),
        rand(rng, MultivariateNormal(Ψ), T)
    )
    u = rand(rng, MultivariateNormal(Σ), T)
    # Generate starting factors
    f₀ = randn(rng, length(μ_f₀)) .* σ_f₀ .+ μ_f₀
    # Initialize return and factor matrices and simulate_economy economy
    r = Matrix{eltype(μ_f₀)}(undef, N, T)
    f = ones(eltype(μ_f₀), K, T+1)
    f[:, 1] = f₀ # Initial factor values
    for t ∈ 1:T
        r[:, t] = B * f[:, t] + u[:, t]
        f[:, t+1] = (I - Φ) * f[:, t] + ϵ[:, t]
    end
    # Add a first row with zero returns for the risk-free asset
    if riskfree_asset
        r = vcat(zeros(eltype(r), 1, T), r)
        S₀ = vcat(1, S₀)
    end
    S = hcat(S₀, S₀ .* exp.(cumsum(r, dims=2)))
    S, r, f # Output prices, returns and factors
end

function calibrate_gp(df::DataFrame; var_mod::Float32 = 1f0)
    # Compute factors and differences thereof
    tickers = sort(unique(df.ticker))
    N = length(tickers)
    df = vcat(
        [add_factors(filter(r -> r.ticker == ticker, df)) for ticker ∈ tickers]...
    )
    S₀ = Float32.(vcat([df.price[df.ticker .== t][1] for t ∈ tickers]...))
    # Feasible Generalized Least Squares estimation
    X = hcat(ones(nrow(df)), Matrix(df[!, [:fW, :fM, :fY]])) # [:fD, :fW, :fM, :fY]])) 
    y = df.ret
    β̂ = FGLS(X, y)
    B = Float32.(kron(β̂', diagm(ones(N))))
    # Compute mean-reversion parameters
    ϕ = [0, df.fW \ df.ΔfW, df.fM \ df.ΔfM, df.fY \ df.ΔfY]
    Φ = Float32.(diagm(kron(ϕ, ones(N))))
    # Compute variance-covariance matrices
    # df[!, :ϵD] = df.ΔfD - ϕ[2] * df.fD
    df[!, :ϵW] = df.ΔfW - ϕ[2] * df.fW
    df[!, :ϵM] = df.ΔfM - ϕ[3] * df.fM
    df[!, :ϵY] = df.ΔfY - ϕ[4] * df.fY
    f = hcat(
        ones(size(df, 1) ÷ N, N),
        # hcat([df.ϵD[df.ticker .== ticker] for ticker ∈ tickers]...),
        hcat([df.ϵW[df.ticker .== ticker] for ticker ∈ tickers]...),
        hcat([df.ϵM[df.ticker .== ticker] for ticker ∈ tickers]...),
        hcat([df.ϵY[df.ticker .== ticker] for ticker ∈ tickers]...)
    )
    μ_f₀ = Float32.(vec(mean(f, dims=1)))
    σ_f₀ = Float32.(vec(std(f, dims=1)))
    Ψ = Float32.(cov(f[:, N+1:end]))
    df[!, :u] = df.ret - X*β̂
    Σ = Float32.(cov(hcat([df.u[df.ticker .== ticker] for ticker ∈ tickers]...))) .* var_mod
    GPSimulator(S₀, μ_f₀, σ_f₀, B, Σ, Ψ, Φ)
end