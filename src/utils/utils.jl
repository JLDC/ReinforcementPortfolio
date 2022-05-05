uniform_weights(n::Int) = ones(Float32, n) / n
ispossemidef(A) = minimum(eigvals(A)) ≥ 0
sharpe_ratio(x) = mean(x) / std(x)

# Feasible Generalized Least Squares esimation
function FGLS(X, y, ϵ=1e-10, max_iter=25)
    β̂_ols = inv(X'X)*X'y
    σ̂₁ = (y - X*β̂_ols).^2
    Ω̂_inv = inv(diagm(σ̂₁))
    β̂_fgls = inv(X'Ω̂_inv*X)*X'Ω̂_inv*y
    for i ∈ 1:max_iter
        σ̂₂ = (y - X*β̂_fgls).^2 
        md = maximum(abs.(σ̂₁ - σ̂₂))
        println("Max diff $md")
        if md < ϵ
            println("Convergence after iteration $i")
            return β̂_fgls
        else
            σ̂₁ = σ̂₂
            Ω̂_inv = inv(diagm(σ̂₁))
            β̂_fgls = inv(X'Ω̂_inv*X)*X'Ω̂_inv*y
        end
    end
    @warn "No convergence reached"
    β̂_fgls
end

function add_factors(df::DataFrame)
    # Add factor columns for past returns / sharpe ratios
    # df[!, :fD] .= 0.0
    df[!, :fW] .= 0.0
    df[!, :fM] .= 0.0
    df[!, :fY] .= 0.0
    for t ∈ 252:size(df, 1)-1
        # df.fD[t+1] = df.ret[t]
        df.fW[t+1] = sharpe_ratio(df.ret[t-4:t])
        df.fM[t+1] = sharpe_ratio(df.ret[t-21:t])
        df.fY[t+1] = sharpe_ratio(df.ret[t-251:t])
    end
    # Add factor differences
    # df[!, :ΔfD] = vcat(0, diff(df.fD))
    df[!, :ΔfW] = vcat(0, diff(df.fW))
    df[!, :ΔfM] = vcat(0, diff(df.fM))
    df[!, :ΔfY] = vcat(0, diff(df.fY))
    df = df[253:end, :] # Drop superfluous observations
end

checknans(gs) = any([any(isnan.(g)) for g ∈ gs])