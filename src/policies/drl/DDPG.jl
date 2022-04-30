mutable struct DDPG{
    BA<:NeuralNetworkApproximator,
    BC<:NeuralNetworkApproximator,
    TA<:NeuralNetworkApproximator,
    TC<:NeuralNetworkApproximator,
    P,
} <: AbstractPolicy
    behavior_actor::BA
    behavior_critic::BC
    target_actor::TA
    target_critic::TC
    γ::Float32
    ρ::Float32
    na::Int
    batchsize::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_freq::Int
    act_limit::Float32
    act_noise::Float32
    update_step::Int
    train_mode::Bool
end

Flux.functor(x::DDPG) = (
    ba = x.behavior_actor,
    bc = x.behavior_critic,
    ta = x.target_actor,
    tc = x.target_critic,
),
y -> begin
    x = @set x.behavior_actor = y.ba
    x = @set x.behavior_critic = y.bc
    x = @set x.target_actor = y.ta
    x = @set x.target_critic = y.tc
    x
end

function DDPG(;
    behavior_actor,
    behavior_critic,
    target_actor,
    target_critic,
    start_policy,
    γ = 0.9f0,
    ρ = 0.995f0,
    na = 1,
    batchsize = 32,
    start_steps = 10000,
    update_after = 1000,
    update_freq = 50,
    act_limit = 1f0,
    act_noise = 5f-2,
    update_step = 0,
    train_mode = true
)
    copyto!(behavior_actor, target_actor)  # force sync
    copyto!(behavior_critic, target_critic)  # force sync
    DDPG(
        behavior_actor,
        behavior_critic,
        target_actor,
        target_critic,
        γ,
        ρ,
        na,
        batchsize,
        start_steps,
        start_policy,
        update_after,
        update_freq,
        act_limit,
        act_noise,
        update_step,
        train_mode
    )
end

function (p::DDPG)(env::SimulatorEnv)
    p.update_step += 1
    if p.update_step ≤ p.start_steps && p.train_mode
        p.start_policy(env)
    else
        D = device(p.behavior_actor)
        s = state(env)
        action = p.behavior_actor(s) |> vec
        if p.train_mode
            clamp.(action .+ randn(p.na) .* repeat([p.act_noise], p.na), -p.act_limit, p.act_limit)
        else
            action
        end
    end
end

# Special case for oracle policies that need to be initialized
function RLBase.update!(
    p::DDPG,
    ::AbstractTrajectory,
    env::SimulatorEnv,
    stage::PreEpisodeStage
)
    p.update_step ≤ p.start_steps && p.start_policy(stage, env)
end


function RLBase.update!(
    p::DDPG,
    traj::CircularArraySARTTrajectory,
    ::SimulatorEnv,
    ::PreActStage,
)
    length(traj) > p.update_after || return
    p.update_step % p.update_freq == 0 || return
    _, batch = sample(traj, BatchSampler{SARTS}(p.batchsize))
    update!(p, batch)
end

function RLBase.update!(p::DDPG, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = batch

    A = p.behavior_actor
    C = p.behavior_critic
    Aₜ = p.target_actor
    Cₜ = p.target_critic

    γ = p.γ
    ρ = p.ρ


    a′ = Aₜ(s′)
    qₜ = Cₜ(vcat(s′, a′)) |> vec
    y = r .+ γ .* (1 .- t) .* qₜ
    a = Flux.unsqueeze(a, ndims(a)+1)

    ∇ = gradient(Flux.params(C)) do
        q = C(vcat(s, a)) |> vec
        loss = mean((y .- q) .^ 2)
        loss
    end

    update!(C, ∇)

    ∇ = gradient(Flux.params(A)) do
        loss = -mean(C(vcat(s, A(s))))
        loss
    end

    update!(A, ∇)

    # polyak averaging
    for (dest, src) in zip(Flux.params([Aₜ, Cₜ]), Flux.params([A, C]))
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end