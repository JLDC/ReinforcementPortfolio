function create_vpg_agent(
    env; 
    γ=0.9f0, 
    hidden_size=64, 
    opt_a=ADAM(), 
    opt_b=ADAM(), 
    batchsize=32,
    activation=relu
)
    na = nassets(env)
    in_size = size(env.f, 1) # This has the weights vector included
    Agent(
        policy = VPG(
            γ = γ,
            batchsize = batchsize,
            approximator = NeuralNetworkApproximator(
                model = GaussianNetwork(
                    μ = Chain(
                        Dense(in_size, hidden_size, activation),
                        Dense(hidden_size, hidden_size, activation),
                        Dense(hidden_size, na, identity)
                    ),
                    logσ = Chain(
                        Dense(in_size, hidden_size, activation),
                        Dense(hidden_size, hidden_size, activation),
                        Dense(hidden_size, na, identity)
                    ),
                    # min_σ = 1f-5,
                    max_σ = 5f-1,
                    normalizer = tanh
                ),

                optimizer = opt_a
            ),
            baseline = NeuralNetworkApproximator(
                model = Chain(
                    Dense(in_size, hidden_size, activation),
                    Dense(hidden_size, hidden_size, activation),
                    Dense(hidden_size, 1, identity)
                ),
                optimizer = opt_b
            )
        ),
        trajectory = VectorSARTTrajectory(
            state = typeof(state(env)),
            action = Vector{Float32}
        )
    )
end