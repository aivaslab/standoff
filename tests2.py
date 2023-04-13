from src.train import main as trainMain
from src.evaluate import main as evalMain
from src.visualize import main as visualizeMain
import sys
import os 

sys.path.append(os.getcwd())
#import torch
#torch.cuda.is_available = lambda : False

print('train')

if True:
    trainMain(['--experiment_name', 'b2-dif1',
            '--env_group', '3',
            '--timesteps', '5e5',
            '--checkpoints', '20',
            '--size', '13',
            '--model_class', 'PPO',
            '--threads', '16',
            '--difficulty', '1',
            '--vecNormalize',
            '--overwrite', 
            '--lr', '1e-4', 
            #'--hidden_size', '32',
            '--n_steps', '2048',
            '--savePath', 'save_dir',
            #'--start_at', '2',
            ])
            
print('evaluate')
            
if True:
    evalMain(['--path', 'save_dir/b2-dif1',
            '--env_group', '4',
            '--make_vids',
            '--make_evals', '--episodes', '20'])
            
print('visualize')
            
visualizeMain(['--path', 'save_dir/b2-dif1'])
