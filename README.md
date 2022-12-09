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
    - Seems lr=0.001, scheduler=100, gamma=0.5 works well
- [x] Sobolev norm loss function
- [x] U-net for baseline comparison
- [x] File cleanup & Code refactor
- [x] Increased resolution testing framework
- [x] Add learning rate scheduler
- [x] Run on high res data
- [ ] Num parameters, time per epoch for all


Final Tests:
- [ ] Fix mod network
- [ ] Alpha: 0.0001, 0.001, 0.01
- [ ] Batch size: 5, 10, 20, 50
- [ ] Get test MSE, train time per epoch
- [ ] Lambda search: don't show
- [ ] Upscaling with 2 networks again

