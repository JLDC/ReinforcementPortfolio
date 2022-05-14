# Portfolio Management

We consider a portfolio management framework where the investor invests her capital among $N$ assets for a total of $T$ time steps.

The investor is not allowed to borrow money or to short assets, and she invests her full capital at all times. This implies that the investment choice can be represented by a vector on the $N$-dimensional unit simplex, i.e.

$$\mathbf{w}_t = [w_1, w_2, \dots, w_N]^\top, \quad \text{with } w_{i,t} \geq 0 \ \forall i, \ \ \sum_{i=1}^N w_{i,t} = 1.$$

Thus, $w_{i,t}$ indicates the proportion of wealth allocated in the $i^\text{th}$ asset at time $t$.

The investor reallocates her portfolio at discrete time steps $t=1, 2, \dots, T$. Between each reallocation, the portfolio weights fluctuate due to the market movements. For instance, if we admit $t$ to be the market close, the portfolio weights at time $t+1$ **before reallocating** are not given by $\mathbf{w}_t$ but rather 

$$\mathbf{w}^\prime_t = \frac{\mathbf{w}_t \odot \mathbf{y}_{t+1}}{\mathbf{w}_t^\top \mathbf{y}_{t+1}},$$

with $\odot$ being the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) and $\mathbf{y}_{t+1}$ the price-relative vector, i.e.

$$\mathbf{y}_{t+1} = \left[\frac{p_{1,t+1}}{p_{1,t}}, \frac{p_{2,t+1}}{p_{2,t}}, \dots, \frac{p_{N,t+1}}{p_{N,t}}\right]^\top,$$

where $p_{i,t}$ is the price of the $i^\text{th}$ asset at time $t$.

When reallocating the portfolio from $\mathbf{w}^\prime_{t-1}$ to $\mathbf{w}_t$, the agent incurs a percentage transaction cost given by

$$\text{TC}(\Delta \mathbf{w}_t) = c \cdot \sum_{i=1}^N | w_{i,t} - w^\prime_{i,t-1}|,$$

where $c$ represents the broker's percentage fee on each trade, e.g., $c=0.01$ amounts to a $1\%$ fee. If the investor has a capital of $C_t$ before rebalancing at time $t$ and decides to rebalance her portfolio to $\mathbf{w}_t$, she incurs a loss of $\text{TC}(\Delta \mathbf{w}_t) \cdot C_t$ due to transaction costs.