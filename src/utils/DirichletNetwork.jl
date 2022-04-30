using Zygote: ignore

Base.@kwdef struct DirichletNetwork{A}
    α::A
    min_α::Float32 = 1f0
    max_α::Float32 = 1f6
end

Flux.@functor DirichletNetwork

function(model::DirichletNetwork)(rng::AbstractRNG, s; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    raw_α = model.α(s)
    α = clamp.(raw_α, model.min_α, model.max_α)
    if is_sampling
        act = mapreduce(x -> rand(rng, Dirichlet(x)), hcat, eachcol(α))
        if is_return_log_prob
            logp_π = mapreduce(x -> logpdf(Dirichlet(x[2]), x[1]), hcat, zip(eachcol(act), eachcol(α)))
            return act, logp_π
        else
            return act
        end
    else
        return α
    end    
end
function(model::DirichletNetwork)(state; is_sampling::Bool=false, is_return_log_prob::Bool=false)
    model(Random.GLOBAL_RNG, state, is_sampling=is_sampling, is_return_log_prob=is_return_log_prob)
end

function (model::DirichletNetwork)(state, action)
    raw_α = model.α(state)
    α = clamp.(raw_α, model.min_α, model.max_α)
    logp_π = mapreduce(x -> clamp.(logpdf(Dirichlet(x[2]), x[1]), hcat, zip(eachcol(action), eachcol(α))))
    logp_π
end