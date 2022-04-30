"""
    BuyAndHold <: AbstractPolicy

Buy and hold strategy. Buy some amount of stock `w` and hold them forever.
"""
mutable struct BuyAndHold <: AbstractPolicy
    w::Vector{Float64}
end
# On first step allocate wealth and then return the current environment portfolio
(p::BuyAndHold)(env::SimulatorEnv) = is_firststep(env) ? p.w : env.w

UniformBuyAndHold(env::SimulatorEnv) = BuyAndHold(uniform_weights(env))