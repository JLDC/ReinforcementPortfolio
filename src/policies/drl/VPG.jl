@Base.kwdef mutable struct VPG{
    A<:NeuralNetworkApproximator,
    B<:NeuralNetworkApproximator,
} <: AbstractPolicy
    approximator::A # Should be a GaussianNetwork
    baseline::B
    γ::Float32 = 0.9f0 # Discount factor
    batchsize::Int = 32
    train_mode::Bool = true
end

# TODO: GPU/CPU agnosticism
function (p::VPG)(env::AbstractEnv)
    if p.train_mode
        p.approximator(state(env); is_sampling=true)
    else
        p.approximator(state(env); is_sampling=false)[1]
    end
end

function RLBase.update!(
    traj::AbstractTrajectory,
    ::VPG,
    env::AbstractEnv,
    ::PreActStage,
    action
)
    push!(traj[:state], state(env))
    push!(traj[:action], action)
end

function RLBase.update!(
    traj::AbstractTrajectory,
    ::VPG,
    ::AbstractEnv,
    ::PreEpisodeStage
)
    empty!(traj)
end

function RLBase.update!(
    ::VPG, 
    ::AbstractTrajectory, 
    ::AbstractEnv, 
    ::PreActStage
)
    nothing
end

# TODO: GPU/CPU agnosticism
function RLBase.update!(
    p::VPG,
    traj::AbstractTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage
)
    p.train_mode || return nothing # Do not change policy in test mode
    m = p.approximator
    b = p.baseline

    states = hcat(traj[:state]...)
    actions = hcat(traj[:action]...)
    gains = traj[:reward] |> x -> discount_rewards(x, p.γ)

    for idx ∈ Iterators.partition(shuffle(1:length(traj[:terminal])), p.batchsize)
        S = states[:, idx]
        A = actions[:, idx]
        G = gains[idx] |> x -> Flux.unsqueeze(x, 1)

        # Update baseline
        ∇ = gradient(Flux.params(b)) do 
            mean((G - b(S)).^ 2)
        end
        update!(b, ∇)
        Â = G - b(S) # Advantage estimate

        # Update policy
        ∇ = gradient(Flux.params(m)) do
            log_prob = m(S, A)
            -mean(Â .* log_prob)
        end
        update!(m, ∇)
    end
end