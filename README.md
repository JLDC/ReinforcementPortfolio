# ReinforcementPortfolio

Set of codes for the paper *Deep Reinforcement Learning for Portfolio Management: A Simulation Study*
## TODO List
### High priority
- [ ] Documentation
- [ ] Example usage
- [ ] Pre-stored `GPEnv` parameters.
- [ ] Generalize `SimulatorEnv` to have the same components: a simulator which generates prices, returns, features. => this can then be used to unify `state()`, `reset!()`, and `env(action)`. Note: must keep track of `returns` for rewards such as SR.
### Medium priority
- [ ] Make agents work for any environment type, not only `GPEnv`
- [ ] Add RNG everywhere instead of resetting seeds within functions? -> Unsure
- [ ] Gradient clipping for PPO
- [ ] CPU/GPU agnosticism
- [ ] Recurrent policies
### Low priority
- [ ] State space for the environment
- [ ] GAIL