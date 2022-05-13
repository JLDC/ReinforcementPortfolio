var documenterSearchIndex = {"docs":
[{"location":"lib/internals/#Internals-Documentation","page":"Internals","title":"Internals Documentation","text":"","category":"section"},{"location":"lib/internals/#Index","page":"Internals","title":"Index","text":"","category":"section"},{"location":"lib/internals/","page":"Internals","title":"Internals","text":"Pages = [\"internals.md\"]","category":"page"},{"location":"lib/internals/#Internals-Interface","page":"Internals","title":"Internals Interface","text":"","category":"section"},{"location":"lib/internals/","page":"Internals","title":"Internals","text":"Modules = [ReinforcementPortfolio]\nPublic = false","category":"page"},{"location":"lib/internals/#ReinforcementPortfolio.GBMSimulator","page":"Internals","title":"ReinforcementPortfolio.GBMSimulator","text":"GBMSimulator\n\nAn asset simulator where the prices follow a geometric brownian motion.\n\n\n\n\n\n","category":"type"},{"location":"lib/internals/#ReinforcementPortfolio.GPSimulator","page":"Internals","title":"ReinforcementPortfolio.GPSimulator","text":"GPSimulator\n\nAn asset simulator where the prices follow the model of Gârleanu & Pedersen (2013).\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#Public-Documentation","page":"Public","title":"Public Documentation","text":"","category":"section"},{"location":"lib/public/#Index","page":"Public","title":"Index","text":"","category":"section"},{"location":"lib/public/","page":"Public","title":"Public","text":"Pages = [\"public.md\"]","category":"page"},{"location":"lib/public/#Public-Interface","page":"Public","title":"Public Interface","text":"","category":"section"},{"location":"lib/public/#Online-Portfolio-Selection-Strategies","page":"Public","title":"Online Portfolio Selection Strategies","text":"","category":"section"},{"location":"lib/public/","page":"Public","title":"Public","text":"BuyAndHold\nUniformBuyAndHold\nConstantlyRebalanced\nUniformCRP\nPAMRPolicy\nPAMR\nPAMR1\nPAMR2","category":"page"},{"location":"lib/public/#ReinforcementPortfolio.BuyAndHold","page":"Public","title":"ReinforcementPortfolio.BuyAndHold","text":"BuyAndHold <: AbstractPolicy\n\nBuy and hold strategy. Buy some amount of stock w and hold them forever.\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.UniformBuyAndHold","page":"Public","title":"ReinforcementPortfolio.UniformBuyAndHold","text":"UniformBuyAndHold(env)\n\nCreates a uniform BuyAndHold strategy for the  given environment env.\n\n\n\n\n\n","category":"function"},{"location":"lib/public/#ReinforcementPortfolio.ConstantlyRebalanced","page":"Public","title":"ReinforcementPortfolio.ConstantlyRebalanced","text":"ConstantlyRebalanced <: AbstractPolicy\n\nConstantly rebalanced portfolio. Reallocates to fixed weights w at chosen frequency freq.\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.UniformCRP","page":"Public","title":"ReinforcementPortfolio.UniformCRP","text":"UniformCRP(env, freq = 30)\n\nCreates a uniform ConstantlyRebalanced strategy for the given environment env and rebalancing frequency freq.\n\n\n\n\n\n","category":"function"},{"location":"lib/public/#ReinforcementPortfolio.PAMRPolicy","page":"Public","title":"ReinforcementPortfolio.PAMRPolicy","text":"PAMRPolicy\n\nAbstract type for Passive Aggressive Mean Reversion strategies.\n\nReference:         B. Li, P. Zhao, S. C.H. Hoi, and V. Gopalkrishnan.         Pamr: Passive aggressive mean reversion strategy for portfolio selection, 2012.         https://link.springer.com/content/pdf/10.1007/s10994-012-5281-z.pdf\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.PAMR","page":"Public","title":"ReinforcementPortfolio.PAMR","text":"PAMR(w, ϵ, freq)\n\nOriginal Passive Aggressive Mean Reversion strategy. See  PAMRPolicy.\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.PAMR1","page":"Public","title":"ReinforcementPortfolio.PAMR1","text":"PAMR1(w, ϵ, C, freq)\n\nFirst variant of the Passive Aggressive Mean Reversion strategy. See  PAMRPolicy.\n\n\n\n\n\n","category":"type"},{"location":"lib/public/#ReinforcementPortfolio.PAMR2","page":"Public","title":"ReinforcementPortfolio.PAMR2","text":"PAMR2(w, ϵ, C, freq)\n\nSecond variant of the Passive Aggressive Mean Reversion strategy. See  PAMRPolicy.\n\n\n\n\n\n","category":"type"},{"location":"lib/public/","page":"Public","title":"Public","text":"<!– ### Optimal Strategies","category":"page"},{"location":"lib/public/","page":"Public","title":"Public","text":"DeterministicPolicy\nOraclePolicy","category":"page"},{"location":"lib/public/","page":"Public","title":"Public","text":"–>","category":"page"},{"location":"#ReinforcementPortfolio.jl","page":"Home","title":"ReinforcementPortfolio.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ReinforcementPortfolio.jl","category":"page"}]
}