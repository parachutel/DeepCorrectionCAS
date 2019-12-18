<pre>          <img src="/docs/Airbus_UTM_lockup.jpg" width="300">          <font size="10"></font>          <img src="/docs/logo.png" height=100> </pre>

# Aircraft Collision Avoidance Systems (CAS) with Deep Corrections


This repository has four main components
- `VICAS`: a package for generating MDP table policies
- `DeepCorrection`: a package for learning deep corrections for MDP table policies
- `CASIM`: a multi-agent free flight airspace simulator
- `Evaluation`: tools for evaluating performance of collision avoidance systems

## Requirements
This project uses Julia v0.6.4. Required packages are listed in `JuliaPkgs.txt`.

## VICAS
Generate MDP table policies by running `julia ./VICAS/src/gen_policy.jl`. State space discretization and state transition sigma sampling schemes are defined in `/VICAS/src/discretizations`. The discrete MDP is solved by value iteration (VI). Notebooks in `/VICAS/viz` are used to interactively visualize policy slices. A sample of pairwise policy visualization is shown below:  
<img src="/docs/vicas_policy.gif" width="250" height="249">


## DeepCorrection
Train deep corrections for MDP table policies (generated by `VICAS`) by running `julia ./DeepCorrection/src/train.jl`. Detailed algorithm implementation can be found in `/DeepCorrection/DeepQLearning.jl/src`. Use `/DeepCorrection/src/viz_policy_multi.jl` to visualize the corrected policy with multiple intruders.  
<img src="/docs/deepcorrection.png" width="534" height="200"><img src="/docs/corrected_closest_policy.gif" width="250" height="249">

## CASIM
This is a package for simulating CAS in a free flight airspace. Various performance metrics are tracked and recorded. Run simulation using files in `CASIM/benchmarking` which specified various scenarios. Simulation animation can be generated by running `CASIM/src/airspace_sim_animation.jl`. A sample animation for simulation is shown below.  
<img src="/docs/casim_sample.gif" width="562" height="300">  
Tracked statistics can be found in `/CASIM/src/Stats_module.jl`. Encounter distribution from the statistics can be visualized using `/CASIM/Encounter Distribution.ipynb`.  
<img src="/docs/encounter_distributions.png" width="553" height="250">

## Evaluation
This folder contains some evaluation tools. 
- `ResolutionRatio`: evaluating the success rate of CAS in resolving encounters
- `ResolutionTime`: evaluating the time CAS use to resolve encounters
- `Sensitivity`: evaluating the alert sensitivity of CAS  
<img src="/docs/sensitivity.png" width="280" height="216">  

- `TrajectoryViz`: visualizing flight trajectories under designed encounter scenarios with selected CAS  

NoCAS | VICASClosest |  CorrectedSector
:-:|:-------------------------:|:-----------------------:
<img src="/docs/nocas_trajviz.gif" width="250" height="250"> | <img src="/docs/vicas_closest_trajviz.gif" width="250" height="250">  |  <img src="/docs/corrected_sector_trajviz.gif" width="250" height="250">



## Publication
S. Li, M. Egorov, and M. J. Kochenderfer, “[Optimizing collision avoidance in dense airspace using deep reinforcement learning](http://www.atmseminarus.org/seminarContent/seminar13/papers/ATM_Seminar_2019_paper_65.pdf),” in *Air Traffic Management Research and Development Seminar*, 2019.
