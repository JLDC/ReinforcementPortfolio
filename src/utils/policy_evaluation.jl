using JLD

function evaluate_policy(
    policy::AbstractPolicy, env::SimulatorEnv, outfile; 
    n_episodes::Int=10_000, seed::Int=72
)
    hook = RewardsPerEpisode()
    Random.seed!(seed)
    reset!(env)
    # Run policy for n_episodes
    run(policy, env, StopAfterEpisode(n_episodes), hook)
    # Store hook
    save(outfile, "rewards", hook.rewards)
end