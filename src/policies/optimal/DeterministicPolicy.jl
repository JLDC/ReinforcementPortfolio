@Base.kwdef mutable struct DeterministicPolicy <: AbstractPolicy
    w::Union{Nothing, AbstractMatrix} = nothing
    verbose::Bool = false
end

function (p::DeterministicPolicy)(::PreEpisodeStage, env::SimulatorEnv)
    @unpack sim, T, fee, w, f = env
    @unpack B, Φ = sim
    @unpack verbose = p
    N = nassets(env)
    Bf = [exp.(B*(I-Φ)^t*f[:, 1]) for t ∈ 1:T]
    pf_model = Model(Ipopt.Optimizer)
    verbose || set_optimizer_attribute(pf_model, "print_level", 0)
    # Portfolio weights over time as matrix (no shorting or leverage)
    @variable(pf_model, 0 <= x[1:N, 1:T] <= 1)
    # Add constraint on simplex for each period
    @constraint(pf_model, [t ∈ 1:T], sum(x[:, t]) == 1)
    # Cost function
    @NLobjective(pf_model, Max,
        log(sum(x[i, 1] * Bf[1][i] for i ∈ 1:N) * (1 - fee * sum(abs(x[i, 1] - w[i]) for i ∈ 1:N))) +
        sum(
            log(sum(x[i, t] * Bf[t][i] for i ∈ 1:N) * (1 - fee * sum(abs(x[i, t] - x[i, t-1]) for i ∈ 1:N)))
            for t ∈ 2:T
        )
    )
    optimize!(pf_model)
    # Update weights with optimal solution
    p.w = value.(x)
end

(p::DeterministicPolicy)(env::SimulatorEnv) = p.w[:, env.t]

function RLBase.update!(
    p::DeterministicPolicy,
    ::AbstractTrajectory,
    env::SimulatorEnv,
    stage::PreEpisodeStage
)
    p(stage, env)
end