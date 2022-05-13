module ReinforcementPortfolio

using ARCHModels
using DataFrames
using Distributions
using Flux
using Ipopt
using JuMP
using LinearAlgebra
using Random
using ReinforcementLearning
using Setfield
using Statistics
using UnPack



export GPEnv, calibrate_gp, GPSVEnv, calibrate_gpsv,
    create_ddpg_agent, create_sac_agent, create_vpg_agent, create_ppo_agent,
    BuyAndHold, UniformBuyAndHold,
    ConstantlyRebalanced, UniformCRP,
    PAMRPolicy, PAMR, PAMR1, PAMR2,
    DeterministicPolicy, OraclePolicy,
    evaluate_policy, 
    ReturnReward, SharpeReward

# Abstract types
include("asset_simulators/AssetSimulator.jl")
include("environments/SimulatorEnv.jl")
# Generic helpers
include("utils/utils.jl")
include("utils/SimplexSpace.jl")
include("utils/policy_evaluation.jl")
include("utils/RewardStyle.jl")
include("utils/CustomLayers.jl")
# Simulators
include("asset_simulators/GBMSimulator.jl")
include("asset_simulators/GPSimulator.jl")
include("asset_simulators/GPSVSimulator.jl")
# Environments
include("environments/GBMEnv.jl")
include("environments/GPEnv.jl")
# include("environments/OracleEnv.jl")
# Policies
#   - OLPS
include("policies/olps/BuyAndHold.jl")
include("policies/olps/ConstantlyRebalanced.jl")
include("policies/olps/PAMR.jl")
#   - DRL
include("policies/drl/VPG.jl")
include("policies/drl/DDPG.jl")
include("policies/drl/SAC.jl")
include("policies/drl/PPO.jl")
#   - Oracle / optimal
include("policies/optimal/DeterministicPolicy.jl")
include("policies/optimal/OraclePolicy.jl")
# Agent builders
include("agents/vpg_agent.jl")
include("agents/ddpg_agent.jl")
include("agents/sac_agent.jl")
include("agents/ppo_agent.jl")
end # module