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


## Getting Started

To run a full experiment, you should run the main functions of three files in sequence: train.py, evaluate.py, and visualize.py. An example of these functions being used together is in tests.py.

### train

train.py is used to train an agent, saving model checkpoints regularly. "--continuing" allows for a run to pick up where another left off. "--repetitions" allows for multiple models to be trained under the same circumstances (experimental, does not work with continuing).

Many hyperparameters are accessible through command line arguments. The "--variable" argument can contain multiple values for a variable, overriding any other values defined by other arguments or defaults. Doing so will train a model for each value and save it in its own subfolder. For example, the following command will produce two subfolders titled "lr0.001" and "lr0.0001" with their own train checkpoints:

```
$python train.py --variable "lr=[0.001,0.0001]"
```

### evaluate 

evaluate.py runs model checkpoints on all the envs of a given env_group (see environment configs below), producing a .csv file with results from each episode. Certain hyperparameters are loaded from a json file saved by train.py to ensure the model and environments use the appropriate configs.

### visualize

visualize.py produces various figures using the csv files produced by evaluate.py.


## Environment configs

The different tasks are referenced in src/pz_envs/scenario_configs.py. Also in that file are env_groups, which are a convenience for referencing multiple tasks (e.g. all the different Stage-3 tasks). Task names are as follows:

```
For stages 1 and 2:
  "Standoff-S{stage}-{difficulty}-{view_size}-{observation_style}-v0"
For stage 3:
  "Standoff-S{stage}-{task_name}-{difficulty}-{view_size}-{observation_style}-v0"
  
"stage" is in [1-3]
"task_name" should have spaces removed.
"difficulty" is in [0-3] (see difficulty levels below)
"view_sizes" are currently in [13, 15, 17, 19]
"observation_style" is in ['rich', 'image']
```

To make a raw environment, you may run any of:

```
gym.make("Standoff-s1-0-17-rich-v0")
gym.make("Standoff-s2-2-19-image-v0")
gym.make("Standoff-s3-removeduninformed-1-15-rich-v0")
```

This repo vectorizes environments using supersuit, using make_env_comp() in src/utils/conversion.py.


## Difficulty levels

The Competitive Feeding paradigm is extremely difficult for RL agents, so we present a ladder of 3 easier variants to attempt before the most difficult challenge. The variants add information to the subject's observation to bypass certain memory and reasoning aspects of the task. They also reveal the subject's movements to the opponent, but the opponent does not act on this information as an animal might. 

|           | Subject decisions  | Persistent treat images | Opponent decisions | Gaze as 4th channel |
|-----------|--------------------|-------------------------|--------------------|---------------------|
| 0      |          ✓         |            ✓            |          ✓         |      Persistent     |
| 1  |          ✓         |            ✓            |          -         |      Persistent     |
| 2 |          ✓         |            -            |          -         |      Transient      |
| 3 |          -         |            -            |          -         |          -          |

"Subject decisions" and "opponent decisions" prevent certain tiles from obscuring movement of the subject and opponent, respectively. "Persistent treat images" allows the subject to view the contents of boxes. "Gaze as 4th Channel" highlights tiles which the opponent sees. If it is persistent, these tiles remain highlighted until they change, even if the opponent stops seeing them.
