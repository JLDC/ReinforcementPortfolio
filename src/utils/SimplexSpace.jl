struct SimplexSpace
    n::Int
end

function Base.in(x::AbstractVector, s::SimplexSpace)
    length(x) == s.n && all(>=(0), x) && isapprox(1, sum(x))
end

function Random.rand(rng::AbstractRNG, s::SimplexSpace)
    x = rand(rng, s.n)
    x ./ sum(x)
end

# Projection onto the simplex
function simplex_proj(x::AbstractVector, b=1)
    x = Float64.(x)
    @assert b > 0 "The radius of the simplex must be positive."
    n = length(x)
    y = sort(x)
    for i ∈ n-1:-1:1
        tᵢ = (sum(y[i+1:end]) - 1) / (n - i)
        tᵢ ≥ y[i] && return Float32.(max.(0, x .- tᵢ))
    end
    Float32.(max.(0, x .- (sum(x) - 1)/n))
end

struct StateSpace end

function Base.in(x, s::StateSpace)
    true
end