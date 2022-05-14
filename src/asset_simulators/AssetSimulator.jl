"""
    AssetSimulator

Abstract type for all asset simulators. An asset simulator incorporates the dynamics which
govern the evolution of asset prices and potential underlying factors.
"""
abstract type AssetSimulator end

"""
    nassets(sim)

Total number of assets for the simulator `sim`.
"""
nassets(sim::AssetSimulator) = length(sim.S₀)