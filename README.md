# Energy-preservation in Operator Learning

This project aims to explore different methods in operator learning for solving PDEs. We have explored two versions based on the DeepONet [[1, 2]](#1), and three versions based on the Fourier Neural Operator (FNO) [[3]](#3).

We also aim to extend the paper on the Energy-consistent Neural Opeator [[4]](#4) by utilizing operator learning methods such as the DeepONet and the FNO in the architecture, and we explore a "hard-constrained approach".
This is done by enforcing a Hamiltonian partial differential structure, 

$$
u_t = \mathcal{G}\frac{\delta \mathcal{H}}{\delta u},
$$

by adding an energy penalty term to the loss function. The final predictions are then taken as either the outputs of the operator network (a "soft-constrained approach"), or through integrating $\mathcal{G}\frac{\delta \mathcal{H}}{\delta u}$ in time (our "hard-constrained approach").
More exploration, particularly when it comes to hyperparameters, is however needed to determine the true potential of this method.

## References
<a id="1">[1]</a> 
Lu Lu, Pengzhan Jin, and George Em Karniadakis. ‘DeepONet: Learning Nonlinear Operators for
Identifying Differential Equations Based on the Universal Approximation Theorem of Operators’.
In: Nature Machine Intelligence 3.3 (Mar. 18, 2021), pp. 218–229. doi: 10.1038/s42256-021-00302-5.

<a id="2">[2]</a>
Sifan Wang, Hanwen Wang, and Paris Perdikaris. ‘Improved Architectures and Training Algorithms
for Deep Operator Networks’. In: Journal of Scientific Computing 92.2 (Aug. 2022), p. 35. doi: 10.1007/
s10915-022-01881-0.

<a id="3">[3]</a>
Nikola Kovachki et al. Neural Operator: Learning Maps Between Function Spaces. May 2, 2024. doi:
10.5555/3648699.3648788. url: http://arxiv.org/abs/2108.08481

<a id="4">[4]</a>
Yusuke Tanaka et al. Neural Operators Meet Energy-based Theory: Operator Learning for Hamiltonian and
Dissipative PDEs. Feb. 14, 2024. url: http://arxiv.org/abs/2402.09018
