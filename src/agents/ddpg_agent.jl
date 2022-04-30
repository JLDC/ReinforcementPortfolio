function create_ddpg_agent(
    env;
    γ=0.9f0,
    hidden_size=64,
    batchsize=32,
    opt_ba=ADAM(),
    opt_ta=ADAM(),
    opt_bc=ADAM(),
    opt_tc=ADAM(),
    activation=relu,
    update_after=1_000,
    start_steps=10_000,
    capacity=1_000_000,
    start_policy=RandomPolicy()
)
    out_size = nassets(env)
    in_size = size(env.f, 1)
    # Actor networks
    behavior_actor = NeuralNetworkApproximator(
        model = Chain(
            Dense(in_size, hidden_size, activation),
            Dense(hidden_size, hidden_size, activation),
            Dense(hidden_size, out_size, identity),
            softmax
        ),
        optimizer = opt_ba
    )
    target_actor = NeuralNetworkApproximator(
        model = Chain(
            Dense(in_size, hidden_size, activation),
            Dense(hidden_size, hidden_size, activation),
            Dense(hidden_size, out_size, identity),
            softmax
        ),
        optimizer = opt_ta
    )
    # Critic networks
    behavior_critic = NeuralNetworkApproximator(
        model = Chain(
            Dense(in_size + nassets(env), hidden_size, activation),
            Dense(hidden_size, hidden_size, activation),
            Dense(hidden_size, 1, identity)
        ),
        optimizer = opt_bc
    )
    target_critic = NeuralNetworkApproximator(
        model = Chain(
            Dense(in_size + nassets(env), hidden_size, activation),
            Dense(hidden_size, hidden_size, activation),
            Dense(hidden_size, 1, identity)
        ),
        optimizer = opt_tc
    )
    Agent(
        policy = DDPG(
            behavior_actor = behavior_actor,
            behavior_critic = behavior_critic,
            target_actor = target_actor,
            target_critic = target_critic,
            start_policy = start_policy,
            γ=γ,
            batchsize=batchsize,
            na=out_size,
            update_after=update_after,
            start_steps=start_steps
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity=capacity,
            state = Vector{Float32} => size(state(env)),
            action = Vector{Float32} => (out_size,)
        )
    )
end