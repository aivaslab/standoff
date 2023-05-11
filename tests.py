from src.train import main as trainMain
from src.evaluate import main as evalMain
from src.visualize import main as visualizeMain
import sys
import os

sys.path.append(os.getcwd())
# import torch
# torch.cuda.is_available = lambda : False

print('train')
all_exp_names = []
curriculum = []
if True:

    experiment_name = '1e6-cur'
    all_exp_names.append(experiment_name)
    curriculum.append(True)
    trainMain(['--experiment_name', experiment_name,
               '--savePath', 'may',
               '--env_group', 's2+s2b',
               '--curriculum', '--pretrain_dir', 's1-1e6',
               '--timesteps', '1e6',
               '--model_class', 'RecurrentPPO',
               '--threads', '16',
               '--overwrite',
               ])

print('evaluate')

for experiment_name, cur in zip(all_exp_names, curriculum):
    if cur:
        evalMain(['--path', 'save_dir3/' + experiment_name,
                  '--env_group', '3',
                  '--curriculum',
                  # '--make_vids',
                  '--make_evals', '--episodes', '100',
                  ])
    else:
        evalMain(['--path', 'save_dir3/' + experiment_name,
                  '--env_group', '3',
                  # '--make_vids',
                  '--make_evals', '--episodes', '100',
                  ])

print('visualize')

for experiment_name, cur in zip(all_exp_names, curriculum):
    if cur:
        visualizeMain(['--path', 'save_dir3/' + experiment_name, '--matrix', '--curriculum'])
    else:
        visualizeMain(['--path', 'save_dir3/' + experiment_name, '--matrix'])
