using Zygote: ignore
# SoftmaxGaussianNetwork
@Base.kwdef struct SoftmaxGaussianNetwork{A}
    f::A
end

Flux.@functor SoftmaxGaussianNetwork

function (m::SoftmaxGaussianNetwork)(rng::AbstractRNG, s; is_sampling::Bool=false)
    μ, σ = m.f(s)
    if is_sampling
        z = ignore() do
            μ .+ randn(rng, Float32, size(μ)) .* σ
        end
        return z
    else
        return μ
    end
end

function (m::SoftmaxGaussianNetwork)(state; is_sampling::Bool=false)
    m(Random.GLOBAL_RNG, state; is_sampling=is_sampling)
end

function (m::SoftmaxGaussianNetwork)(state, action)
    μ, σ = m.f(state)
    sum(normlogpdf(μ, σ, action), dims=1)
end