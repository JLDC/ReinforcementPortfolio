using Documenter
using ReinforcementPortfolio

makedocs(
    sitename = "ReinforcementPortfolio",
    format = Documenter.HTML(),
    modules = [ReinforcementPortfolio],
    pages = [
        "Home" => "index.md",
        "Portfolio Management" => "pm/index.md",
        "Simulators and Environments" => [
            "Asset Simulators" => "env/asset_sim.md",
            "Simulator Environment" => "env/sim_env.md"
        ],
        # "Library" => [
        #     "Public" => "lib/public.md",
        #     "Internals" => "lib/internals.md" 
        # ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JLDC/ReinforcementPortfolio"
)