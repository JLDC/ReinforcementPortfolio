abstract type RewardStyle end

struct ReturnReward <: RewardStyle end
struct SharpeReward <: RewardStyle end