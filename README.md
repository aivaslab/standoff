# The Standoff Environment

<img src="https://github.com/aivaslab/standoff/blob/main/images/compfeed.png" width="250"> <img src="https://github.com/aivaslab/standoff/blob/main/images/compfeed2.png" width="250">

Gridworld environment for competitive-feeding-like theory of mind experiments, based on [MarlGrid](https://github.com/kandouss/marlgrid), a fork of [MiniGrid](https://github.com/Farama-Foundation/gym-minigrid).

## The Competitive Feeding Task

Standoff is a reinforcement learning environment intended to computationally replicate the competitive feeding paradigm, a test designed to evaluate the social cognition skills of animals. It is divided into 3 stages:

* **Stage 1**: The subject may learn to locate treats, which are hidden in boxes.

* **Stage 2**: A conspecific agent will seek the largest treat of which it is aware, and will not (typically) share with the subject.

* **Stage 3**: Presents eight different scenarios in which the conspecific agent's view is temporarily blocked while the treats are shuffled around. 

For the subject to perform well across all the Stage 3 scenarios, it must act on an understanding of the other agent's *beliefs*.

Stage 3 scenarios include: *informedControl, partiallyUninformed, removedInformed, removedUninformed, moved, replaced, misinformed, swapped* and a ninth task *all* which randomly selects from all eight others.


## Difficulty levels

The Competitive Feeding paradigm is extremely difficult for RL agents, so we present a ladder of 3 easier variants to attempt before the most difficult challenge. The variants add information to the subject's observation to bypass certain memory and reasoning aspects of the task.

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

gym.make("Standoff-s2-moderate-v0")

gym.make("Standoff-s3-removedUninformed-difficult-v0")
```

A barebones colab notebook, which trains an RL agent on one variant and evaluates on multiple, can be found [here]() (TBD).
