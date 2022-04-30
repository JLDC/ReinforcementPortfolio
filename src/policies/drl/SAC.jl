mutable struct SAC{
    BA<:NeuralNetworkApproximator,
    BC1<:NeuralNetworkApproximator,
    BC2<:NeuralNetworkApproximator,
    P,
} <: AbstractPolicy
    policy::BA
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    γ::Float32
    τ::Float32
    α::Float32
    batchsize::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_freq::Int
    target_entropy::Float32
    update_step::Int
    train_mode::Bool
end

function SAC(;
    policy,
    qnetwork1,
    qnetwork2,
    target_qnetwork1=deepcopy(qnetwork1),
    target_qnetwork2=deepcopy(qnetwork2),
    γ=0.9f0,
    τ=0.005f0,
    α=0.2f0,
    batchsize=32,
    start_steps=10000,
    update_after=1000,
    update_freq=50,
    action_dims=0,
    update_step=0,
    start_policy=update_step == 0 ? identity : policy,
    train_mode=true
)
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    SAC(
        policy,
        qnetwork1,
        qnetwork2,
        target_qnetwork1,
        target_qnetwork2,
        γ,
        τ,
        α,
        batchsize,
        start_steps,
        start_policy,
        update_after,
        update_freq,
        Float32(-action_dims),
        update_step,
        train_mode
    )
end

function (p::SAC)(env::SimulatorEnv)
    p.update_step += 1

    if p.update_step <= p.start_steps
        p.start_policy(env)
    else
        D = device(p.policy)
        s = send_to_device(D, state(env))
        if p.train_mode # Training mode
            action = p.policy(s; is_sampling=true)
        else # Test mode
            action = p.policy(s)[1]
        end
        send_to_host(action)
    end
end

function RLBase.update!(
    p::SAC,
    traj::CircularArraySARTTrajectory,
    ::SimulatorEnv,
    ::PreActStage,
)
    length(traj) > p.update_after || return
    p.update_step % p.update_freq == 0 || return
    inds, batch = sample(traj, BatchSampler{SARTS}(p.batchsize))
    update!(p, batch)
end

function RLBase.update!(p::SAC, batch::NamedTuple{SARTS})
    p.train_mode || return nothing # Exit if we are not in train mode
    s, a, r, t, s′ = send_to_device(device(p.qnetwork1), batch)

    γ, τ, α = p.γ, p.τ, p.α

    a′, log_π = p.policy(s′; is_sampling=true, is_return_log_prob=true)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* vec(q′ .- α .* log_π)

    # Train Q Networks
    q_input = vcat(s, a)

    q_grad_1 = gradient(Flux.params(p.qnetwork1)) do
        q1 = p.qnetwork1(q_input) |> vec
        Flux.mse(q1, y)
    end
    update!(p.qnetwork1, q_grad_1)
    q_grad_2 = gradient(Flux.params(p.qnetwork2)) do
        q2 = p.qnetwork2(q_input) |> vec
        Flux.mse(q2, y)
    end
    update!(p.qnetwork2, q_grad_2)

    # Train Policy
    p_grad = gradient(Flux.params(p.policy)) do
        a, log_π = p.policy(s; is_sampling=true, is_return_log_prob=true)
        q_input = vcat(s, a)
        q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
        reward = mean(q)
        entropy = mean(log_π)
        α * entropy - reward
    end
    update!(p.policy, p_grad)

    # polyak averaging
    for (dest, src) in zip(
        Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
        Flux.params([p.qnetwork1, p.qnetwork2]),
    )
        dest .= (1 - τ) .* dest .+ τ .* src
    end
end