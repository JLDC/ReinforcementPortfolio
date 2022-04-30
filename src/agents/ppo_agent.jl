function create_ppo_agent(
    env;
    γ::Float32=.9f0,
    ϵ::Float32=.2f0,
    hidden_size::Int=64,
    opt_a=ADAM(),
    opt_b=ADAM(),
    batchsize::Int=32,
    activation=relu
)
    na = nassets(env)
    in_size = size(env.f ,1)
    Agent(
        policy = PPO(
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
            ),
            batchsize = batchsize,
            γ = γ,
            ϵ = ϵ
        ),
        # PPO has a special trajectory due to action log probability being stored
        trajectory = VectorSARTTrajectory(
            state = typeof(state(env)),
            action = Vector{Float32}
        )
    )
end