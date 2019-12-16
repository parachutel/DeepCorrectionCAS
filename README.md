# Aircraft Collision Avoidance Systems (CAS) with Deep Corrections

This repository has four main components
- `VICAS`: a package for generating MDP table policies
- `DeepCorrection`: a package for learning deep corrections for MDP table policies
- `CASIM`: a multi-agent free flight airspace simulator
- `Evaluation`: tools for evaluating performance of collision avoidance systems

## Requirements
This project uses Julia v0.6.4. Required packages are listed in `JuliaPkgs.txt`.

## VICAS
Generate MDP table policies by running `julia ./VICAS/src/gen_policy.jl`. State space discretization and state transition sigma sampling schemes are defined in `/VICAS/src/discretizations`. The discrete MDP is solved by value iteration (VI). Notebooks in `/VICAS/viz` are used to interactively visualize policy slices.

## DeepCorrection
Train deep corrections for MDP table policies (generated by `VICAS`) by running `julia ./DeepCorrection/src/train.jl`. Detailed algorithm implementation can be found in `/DeepCorrection/DeepQLearning.jl/src`. 

## CASIM
This is a package for simulating CAS in a free flight airspace. Various performance metrics are tracked and recorded. Run simulation using files in `CASIM/benchmarking` which specified various scenarios. Simulation animation can be generated by running `CASIM/src/airspace_sim_animation.jl`. 
A sample animation for simulation is shown below:
![Sample Simulation Animation](/docs/casim_sample.gif =500x)

## Evaluation
This folder contains some evaluation tools. 
- `ResolutionRatio`: evaluating the success rate of CAS in resolving encounters
- `ResolutionTime`: evaluating the time CAS use to resolve encounters
- `Sensitivity`: evaluating the alert sensitivity of CAS
- `TrajectoryViz`: visualizing flight trajectories under designed encounter scenarios with selected CAS


## Publication
S. Li, M. Egorov, and M. J. Kochenderfer, “[Optimizing collision avoidance in dense airspace using deep reinforcement learning](http://www.atmseminarus.org/seminarContent/seminar13/papers/ATM_Seminar_2019_paper_65.pdf),” in *Air Traffic Management Research and Development Seminar*, 2019.
