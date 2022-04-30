@Base.kwdef mutable struct PPO{
    A<:NeuralNetworkApproximator,
    B<:NeuralNetworkApproximator
} <: AbstractPolicy
    approximator::A         # Actor should be a Gaussian policy
    baseline::B             # Critic
    ϵ::Float32 = .2f0       # Clipping range
    γ::Float32 = .9f0      # Discount factor
    batchsize::Int = 32          
    train_mode::Bool = true
end

function (p::PPO)(env::AbstractEnv)
    if p.train_mode
        p.approximator(state(env); is_sampling=true)
    else
        p.approximator(state(env); is_sampling=false)[1]
    end
end

function RLBase.update!(
    traj::AbstractTrajectory,
    ::PPO,
    ::AbstractEnv,
    ::PreEpisodeStage
)
    empty!(traj)
end

function RLBase.update!(
    p::PPO,
    traj::AbstractTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage
)
    p.train_mode || return nothing # Do not adjust policy in test mode
    m = p.approximator
    b = p.baseline

    states = hcat(traj[:state]...)
    actions = hcat(traj[:action]...)
    gains = traj[:reward] |> x -> discount_rewards(x, p.γ) |> x -> Flux.unsqueeze(x, 1)
    logp_old = m(states, actions)
    i = 0
    for idx ∈ Iterators.partition(shuffle(1:length(traj[:terminal])), p.batchsize)
        i += 1
        S = states[:, idx]
        A = actions[:, idx]
        G = gains[:, idx]
        logp = logp_old[:, idx]

        # Update critic
        ∇ = gradient(Flux.params(b)) do
            mean((G .- b(S)).^2)
        end
        # clip_gradients!(∇, 1f-3)
        update!(b, ∇)
        Â = G - b(S) # Compute advantage estimate
        # Update actor
        ∇ = gradient(Flux.params(m)) do
            logp′ = m(S, A)
            ratio = exp.(logp′ .- logp)
            surr₁ = ratio .* Â
            surr₂ = clamp.(ratio, 1f0 - p.ϵ, 1f0 + p.ϵ) .* Â
            -mean(min.(surr₁, surr₂))
        end
        update!(m, ∇)
    end
end

function clip_gradients!(∇, ϵ)
    map(x -> clamp!(x, -ϵ, ϵ), ∇)
    nothing
end