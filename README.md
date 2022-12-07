# CS 230 Project
In this project we attempt to use a Fourier Neural Operator to Resolve Shock Conditions in the Euler System of PDEs.

## Repository Structure (TODO)

## To-Do List
- [x] Test FNO archcitecture on unrapped 3-variable Euler solutions
- [x] Test FNO architecture with 2 added input/output channels, for the multivariable problem
- [x] Normalize input/output values
- [x] Implement decoder for output rescaling
- [ ] Parallel Fourier channels for different variables
- [ ] Hyperparameter exploration
- [x] Sobolev norm loss function
- [x] U-net for baseline comparison
- [x] File cleanup & Code refactor
- [x] Increased resolution testing framework
- [x] Add learning rate scheduler
- [ ] Run on high res data
- [ ] Num parameters, time per epoch for all