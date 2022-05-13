mutable struct GPSVSimulator{T<:AbstractFloat}
    S₀::AbstractVector{T}
    μ_f₀::AbstractVector{T}
    σ_f₀::AbstractVector{T}
    B::AbstractMatrix{T}
    Ψ::AbstractMatrix{T}
    Φ::AbstractMatrix{T}
    dcc
    function GPSVSimulator(S₀, μ_f₀, σ_f₀, B, Ψ, Φ, dcc)
        @assert ispossemidef(Ψ) "Ψ is not a valid covariance matrix" 
        new{eltype(μ_f₀)}(S₀, μ_f₀, σ_f₀, B, Ψ, Φ, dcc)
    end
end

nfactors(sim::GPSVSimulator) = length(sim.μ_f₀)
nassets(sim::GPSVSimulator) = length(sim.S₀)

function calibrate_gpsv(df::DataFrame)
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
    # Calibrate DCC on residuals
    dcc = fit(DCC{1, 1, GARCH{1, 1}}, 
        hcat([df.u[df.ticker .== ticker] for ticker ∈ tickers]...), meanspec=NoIntercept)
    GPSVSimulator(S₀, μ_f₀, σ_f₀, B, Ψ, Φ, dcc)
end

function simulate_economy(
    sim::GPSVSimulator, T::Int=500; 
    rng::Union{AbstractRNG,Nothing}=nothing
)
    rng = isnothing(rng) ? Random.GLOBAL_RNG : rng
    @assert T > 1 "T must be larger than 1"
    @unpack S₀, μ_f₀, σ_f₀, B, Ψ, Φ, dcc = sim
    N, K = size(B) # Extract number of assets / factors
    # Generate factor noise
    ϵ = vcat(
        zeros(eltype(μ_f₀), N, T),
        rand(rng, MultivariateNormal(Ψ), T)
    )
    # Generate asset returns noise
    u = permutedims(simulate(dcc, T).data)
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
    S = hcat(S₀, S₀ .* exp.(cumsum(r, dims=2)))
    S, r, f # Output returns and factors
end