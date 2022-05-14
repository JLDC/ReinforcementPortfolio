var documenterSearchIndex = {"docs":
[{"location":"env/asset_sim/#Asset-Simulators","page":"Asset Simulators","title":"Asset Simulators","text":"","category":"section"},{"location":"env/asset_sim/","page":"Asset Simulators","title":"Asset Simulators","text":"The abstract supertype AssetSimulator encompasses all simulators that can be used to generate the asset prices and underlying factors.","category":"page"},{"location":"env/asset_sim/","page":"Asset Simulators","title":"Asset Simulators","text":"ReinforcementPortfolio.AssetSimulator","category":"page"},{"location":"env/asset_sim/#ReinforcementPortfolio.AssetSimulator","page":"Asset Simulators","title":"ReinforcementPortfolio.AssetSimulator","text":"AssetSimulator\n\nAbstract type for all asset simulators. An asset simulator incorporates the dynamics which govern the evolution of asset prices and potential underlying factors.\n\n\n\n\n\n","category":"type"},{"location":"env/asset_sim/","page":"Asset Simulators","title":"Asset Simulators","text":"Each simulator has its own asset returns dynamics, these are described in the docstring of the relevant simulator (see Simulators).","category":"page"},{"location":"env/asset_sim/#Usage","page":"Asset Simulators","title":"Usage","text":"","category":"section"},{"location":"env/asset_sim/","page":"Asset Simulators","title":"Asset Simulators","text":"In general, an AssetSimulator is passed as an input to construct a RL environment of type SimulatorEnv.","category":"page"},{"location":"env/asset_sim/","page":"Asset Simulators","title":"Asset Simulators","text":"However, it is also possible to interact directly with the AssetSimulator object. In particular, the simulate_economy function can be called with the simulator as input to generate a tuple of (asset prices, asset returns, factors):","category":"page"},{"location":"env/asset_sim/","page":"Asset Simulators","title":"Asset Simulators","text":"simulate_economy","category":"page"},{"location":"env/asset_sim/#ReinforcementPortfolio.simulate_economy","page":"Asset Simulators","title":"ReinforcementPortfolio.simulate_economy","text":"simulate_economy(sim, T = 500; rng = nothing)\n\nSimulates T periods of an economy according to the dynamics of the simulator sim.\n\n\n\n\n\n","category":"function"},{"location":"env/asset_sim/#Simulators","page":"Asset Simulators","title":"Simulators","text":"","category":"section"},{"location":"env/asset_sim/#GPSimulator","page":"Asset Simulators","title":"GPSimulator","text":"","category":"section"},{"location":"env/asset_sim/","page":"Asset Simulators","title":"Asset Simulators","text":"GPSimulator","category":"page"},{"location":"env/asset_sim/#ReinforcementPortfolio.GPSimulator","page":"Asset Simulators","title":"ReinforcementPortfolio.GPSimulator","text":"GPSimulator(S₀, μ_f₀, σ_f₀, B, Σ, Ψ, Φ)\n\nAn asset simulator where the prices follow a model similar to that of Gârleanu & Pedersen  (2013)\n\nThe asset log returns adopt the following multidimensional factor model with N assets  and K factors:\n\nbeginaligned\nmathbfr_t+1 = mathbfBmathbff_t + mathbfu_t+1 \nmathbff_t+1 = left(mathbfI - boldsymbolPhiright) mathbff_t + boldsymbolepsilon_t+1\nendaligned\n\nwith\n\nmathbfB in mathbbR^N times K the matrix of factor loadings.\nboldsymbolPhi in mathbbR^K times K the matrix of mean-reversion coefficients \nmathbfu_t sim mathcalN(mathbf0 boldsymbolSigma) the returns innovation term\nboldsymbolepsilon_t sim mathcalN(mathbf0 boldsymbolPsi) the factors innovation term\n\nWhen an economy is simulated according to the GPSimulator, the initial factors  mathbff_0 are randomized according to a normal distribution with mean μ_f₀ and standard deviation σ_f₀. It is assumed that the initial factor distribution is made of independent normal distributions, i.e. σ_f₀ is a vector of standard deviations and not the covariance matrix (it is the diagonal of the covariance matrix).\n\n\n\n\n\n","category":"type"},{"location":"lib/internals/#Internals-Documentation","page":"Internals Documentation","title":"Internals Documentation","text":"","category":"section"},{"location":"lib/internals/#Index","page":"Internals Documentation","title":"Index","text":"","category":"section"},{"location":"lib/internals/","page":"Internals Documentation","title":"Internals Documentation","text":"Pages = [\"internals.md\"]","category":"page"},{"location":"lib/internals/#Internals-Interface","page":"Internals Documentation","title":"Internals Interface","text":"","category":"section"},{"location":"lib/internals/#Asset-Simulators","page":"Internals Documentation","title":"Asset Simulators","text":"","category":"section"},{"location":"env/sim_env/#Simulator-Environment","page":"Simulator Environment","title":"Simulator Environment","text":"","category":"section"},{"location":"env/sim_env/","page":"Simulator Environment","title":"Simulator Environment","text":"The SimulatorEnv integrates AssetSimulator into the generic abstract environment type AbstractEnv from ReinforcementLearning.jl.","category":"page"},{"location":"env/sim_env/","page":"Simulator Environment","title":"Simulator Environment","text":"SimulatorEnv","category":"page"},{"location":"env/sim_env/#ReinforcementPortfolio.SimulatorEnv","page":"Simulator Environment","title":"ReinforcementPortfolio.SimulatorEnv","text":"SimulatorEnv(sim, T, fee, reward_style) <: AbstractEnv\n\nA simulator environment, subtype of the AbstractEnv from the ReinforcementLearning.jl  package.\n\nArguments\n\nsim: An AssetSimulator which governs the asset returns dynamics of the SimulatorEnv.\nT: The total timesteps until an episode finishes.\nfee: The broker's percentage fee on each trade.\nreward_style: The type of reward the agent receives when interacting with the environment.\n\nSee RewardStyle\n\n\n\n\n\n","category":"type"},{"location":"pm/#Portfolio-Management","page":"Portfolio Management","title":"Portfolio Management","text":"","category":"section"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"We consider a portfolio management framework where the investor invests her capital among N assets for a total of T time steps.","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"The investor is not allowed to borrow money or to short assets, and she invests her full capital at all times. This implies that the investment choice can be represented by a vector on the N-dimensional unit simplex, i.e.","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"mathbfw_t = w_1 w_2 dots w_N^top quad textwith  w_it geq 0  forall i   sum_i=1^N w_it = 1","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"Thus, w_it indicates the proportion of wealth allocated in the i^textth asset at time t.","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"The investor reallocates her portfolio at discrete time steps t=1 2 dots T. Between each reallocation, the portfolio weights fluctuate due to the market movements. For instance, if we admit t to be the market close, the portfolio weights at time t+1 before reallocating are not given by mathbfw_t but rather ","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"mathbfw^prime_t = fracmathbfw_t odot mathbfy_t+1mathbfw_t^top mathbfy_t+1","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"with odot being the Hadamard product and mathbfy_t+1 the price-relative vector, i.e.","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"mathbfy_t+1 = leftfracp_1t+1p_1t fracp_2t+1p_2t dots fracp_Nt+1p_Ntright^top","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"where p_it is the price of the i^textth asset at time t.","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"When reallocating the portfolio from mathbfw^prime_t-1 to mathbfw_t, the agent incurs a percentage transaction cost given by","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"textTC(Delta mathbfw_t) = c cdot sum_i=1^N  w_it - w^prime_it-1","category":"page"},{"location":"pm/","page":"Portfolio Management","title":"Portfolio Management","text":"where c represents the broker's percentage fee on each trade, e.g., c=001 amounts to a 1 fee. If the investor has a capital of C_t before rebalancing at time t and decides to rebalance her portfolio to mathbfw_t, she incurs a loss of textTC(Delta mathbfw_t) cdot C_t due to transaction costs.","category":"page"},{"location":"lib/public/#Public-Documentation","page":"Public Documentation","title":"Public Documentation","text":"","category":"section"},{"location":"lib/public/#Index","page":"Public Documentation","title":"Index","text":"","category":"section"},{"location":"lib/public/","page":"Public Documentation","title":"Public Documentation","text":"Pages = [\"public.md\"]","category":"page"},{"location":"lib/public/#Public-Interface","page":"Public Documentation","title":"Public Interface","text":"","category":"section"},{"location":"lib/public/","page":"Public Documentation","title":"Public Documentation","text":"RewardStyle","category":"page"},{"location":"lib/public/#Online-Portfolio-Selection-Strategies","page":"Public Documentation","title":"Online Portfolio Selection Strategies","text":"","category":"section"},{"location":"lib/public/","page":"Public Documentation","title":"Public Documentation","text":"BuyAndHold\nUniformBuyAndHold\nConstantlyRebalanced\nUniformCRP\nPAMRPolicy\nPAMR\nPAMR1\nPAMR2","category":"page"},{"location":"lib/public/#ReinforcementPortfolio.BuyAndHold","page":"Public Documentation","title":"ReinforcementPortfolio.BuyAndHold","text":"BuyAndHold <: AbstractPolicy\n\nBuy and hold strategy. Buy some amount of stock w and hold them forever.\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.UniformBuyAndHold","page":"Public Documentation","title":"ReinforcementPortfolio.UniformBuyAndHold","text":"UniformBuyAndHold(env)\n\nCreates a uniform BuyAndHold strategy for the  given environment env.\n\n\n\n\n\n","category":"function"},{"location":"lib/public/#ReinforcementPortfolio.ConstantlyRebalanced","page":"Public Documentation","title":"ReinforcementPortfolio.ConstantlyRebalanced","text":"ConstantlyRebalanced <: AbstractPolicy\n\nConstantly rebalanced portfolio. Reallocates to fixed weights w at chosen frequency freq.\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.UniformCRP","page":"Public Documentation","title":"ReinforcementPortfolio.UniformCRP","text":"UniformCRP(env, freq = 30)\n\nCreates a uniform ConstantlyRebalanced strategy for the given environment env and rebalancing frequency freq.\n\n\n\n\n\n","category":"function"},{"location":"lib/public/#ReinforcementPortfolio.PAMRPolicy","page":"Public Documentation","title":"ReinforcementPortfolio.PAMRPolicy","text":"PAMRPolicy\n\nAbstract type for Passive Aggressive Mean Reversion strategies.\n\nReference:         B. Li, P. Zhao, S. C.H. Hoi, and V. Gopalkrishnan.         Pamr: Passive aggressive mean reversion strategy for portfolio selection, 2012.         https://link.springer.com/content/pdf/10.1007/s10994-012-5281-z.pdf\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.PAMR","page":"Public Documentation","title":"ReinforcementPortfolio.PAMR","text":"PAMR(w, ϵ, freq = 30)\n\nOriginal Passive Aggressive Mean Reversion strategy. See  PAMRPolicy.\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.PAMR1","page":"Public Documentation","title":"ReinforcementPortfolio.PAMR1","text":"PAMR1(w, ϵ, C, freq = 30)\n\nFirst variant of the Passive Aggressive Mean Reversion strategy. See  PAMRPolicy.\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.PAMR2","page":"Public Documentation","title":"ReinforcementPortfolio.PAMR2","text":"PAMR2(w, ϵ, C, freq = 30)\n\nSecond variant of the Passive Aggressive Mean Reversion strategy. See  PAMRPolicy.\n\n\n\n\n\n","category":"type"},{"location":"#ReinforcementPortfolio.jl","page":"Home","title":"ReinforcementPortfolio.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ReinforcementPortfolio.jl provides a framework to apply deep reinforcement learning methods to portfolio management. The reinforcement learning components are implemented through the ReinforcementLearning.jl package to make them as general as possible.","category":"page"}]
}
