using Ipopt
using JuMP
using LinearAlgebra
using UnPack

@Base.kwdef mutable struct LQCPolicy <: AbstractPolicy
    w::Union{Nothing, AbstractMatrix} = nothing
    verbose::Bool = false
end

function (p::LQCPolicy)(::PreEpisodeStage, env::SimulatorEnv)
    @unpack sim, T, fee, w, f = env
    @unpack B, Φ, Σ, Ψ = sim
    @unpack verbose, γ = p
    N = nassets(env)
    Λ = diagm([fee for _ ∈ 1:N])
    pf_model = Model(Ipopt.Optimizer)
    verbose || set_optimizer_attribute(pf_model, "print_level", 0)
    # Portfolio weights over time as matrix (no shorting or leverage)
    @variable(pf_model, 0 <= x[1:N, 1:T] <= 1)
    # Add constraint on simplex for each period
    @constraint(pf_model, [t ∈ 1:T], sum(x[:, t]) == 1)
    # Cost function
    @objective(pf_model, Max,
        x[:, 1]'B*f[:, 1] - .5*(x[:, 1]-w)'Λ*(x[:, 1]-w) +
        sum((
            x[:, t]'B*f[:, t+1] 
            - .5*(x[:, t]-x[:, t-1])'Λ*(x[:, t]-x[:, t-1])) 
        for t ∈ 2:T)
    )
    optimize!(pf_model)
    # Update weights with optimal solution
    p.w = value.(x)
end

(p::LQCPolicy)(env::SimulatorEnv) = p.w[:, env.t]

function RLBase.update!(
    p::LQCPolicy,
    ::AbstractTrajectory,
    env::SimulatorEnv,
    stage::PreEpisodeStage
)
    p(stage, env)
end