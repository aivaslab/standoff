# The Standoff Environment

<img src="https://github.com/aivaslab/standoff/blob/main/images/compfeed.png" width="250"> <img src="https://github.com/aivaslab/standoff/blob/main/images/compfeed2.png" width="250">

Gridworld environment for competitive-feeding-like theory of mind experiments, based on [MarlGrid](https://github.com/kandouss/marlgrid), a fork of [MiniGrid](https://github.com/Farama-Foundation/gym-minigrid).

## The Competitive Feeding Task

Standoff computationally replicates the competitive feeding paradigm, a test designed to evaluate the social cognition skills of animals. It is divided into 3 stages:

* **Stage 1**: The subject learns to locate treats, which are hidden in boxes.

* **Stage 2**: The subject learns to share the environment with another agent which will seek out the largest treat it has seen and will not share with the subject.

* **Stage 3**: The subject is able to observe that the other agent's view may be temporarily blocked while the treats are shuffled around. 

There are 8 different Stage 3 scenarios, featuring different kinds of treat movement and different view obscuration timings. For the subject to perform well across all the Stage 3 scenarios, it must remember and act on what the other agent was able to see.

Stage 3 scenarios include: *informedControl, partiallyUninformed, removedInformed, removedUninformed, moved, replaced, misinformed, swapped* and a special ninth setup *all* which randomly selects from the eight others each episode. 


## Difficulty levels

The Competitive Feeding paradigm is extremely difficult for RL agents, so we present a ladder of 3 easier variants to attempt before the most difficult challenge. The variants add information to the subject's observation to bypass certain memory and reasoning aspects of the task. They also reveal the subject's movements to the opponent, but the opponent does not act on this information as an animal might.

|           | Subject decisions  | Persistent treat images | Opponent decisions | Gaze as 4th channel |
|-----------|--------------------|-------------------------|--------------------|---------------------|
| Easy      |          ✓         |            ✓            |          ✓         |      Persistent     |
| Moderate  |          ✓         |            ✓            |          -         |      Persistent     |
| Difficult |          ✓         |            -            |          -         |      Transient      |
| Challenge |          -         |            -            |          -         |          -          |


## Getting Started

Environments are instantiated as follows:

```
gym.make("Standoff-s1-easy-v0")

gym.make("Standoff-s2-difficult-v0")

gym.make("Standoff-s3-removedUninformed-moderate-v0")
```

A barebones colab notebook, which trains an RL agent on one variant and evaluates on multiple, can be found [here]() (TBD).
