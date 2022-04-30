function create_sac_agent(
    env;
    γ=0.9f0,
    hidden_size=64,
    batchsize=32,
    opt_p=ADAM(),
    opt_q1=ADAM(),
    opt_q2=ADAM(),
    activation=relu
)
    out_size = nassets(env)
    in_size = size(env.f, 1)
    # Policy network
    policy_net = NeuralNetworkApproximator(
        model = GaussianNetwork(
            μ = Chain(
                Dense(in_size, hidden_size, activation),
                Dense(hidden_size, hidden_size, activation),
                Dense(hidden_size, out_size, identity)
            ),
            logσ = Chain(
                Dense(in_size, hidden_size, activation),
                Dense(hidden_size, hidden_size, activation),
                Dense(hidden_size, out_size, identity)
            ),
            # min_σ = 1f-5,
            max_σ = 5f-1,
            normalizer = tanh
        ),
        optimizer = opt_p
    )
    # Q-networks
    q_net1 = NeuralNetworkApproximator(
        model = Chain(
            Dense(in_size + nassets(env), hidden_size, activation),
            Dense(hidden_size, hidden_size, activation),
            Dense(hidden_size, 1)
        ),
        optimizer = opt_q1
    )
    q_net2 = NeuralNetworkApproximator(
        model = Chain(
            Dense(in_size + nassets(env), hidden_size, activation),
            Dense(hidden_size, hidden_size, activation),
            Dense(hidden_size, 1)
        ),
        optimizer = opt_q2
    )
    Agent(
        policy = SAC(
            policy = policy_net,
            qnetwork1 = q_net1, 
            qnetwork2 = q_net2,
            action_dims = out_size,
            start_policy = UniformBuyAndHold(env),
            γ=γ,
            batchsize=batchsize
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1_000_000,
            state = Vector{Float32} => size(state(env)),
            action = Vector{Float32} => (out_size,)
        )
    )
end