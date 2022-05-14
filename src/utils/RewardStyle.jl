"""
    RewardStyle

The type of rewards received by an agent when interacting with an environment.
"""
abstract type RewardStyle end

struct ReturnReward <: RewardStyle end
struct SharpeReward <: RewardStyle end