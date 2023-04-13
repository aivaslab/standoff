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
    trainMain(['--experiment_name', 'redo4',
            '--env_group', '5',
            '--timesteps', '1e5',
            '--checkpoints', '20',
            '--size', '13',
            '--model_class', 'PPO',
            '--threads', '16',
            '--difficulty', '0',
            '--vecNormalize',
            '--overwrite', 
            '--lr', '1e-4', 
            #'--hidden_size', '32',
            '--n_steps', '1024',
            '--savePath', 'save_dir',
            '--variable', 'batch_size=[32, 64, 128, 256]'
            #'--start_at', '0',
            #'--end_at', '2',
            ])
            
print('evaluate')
            
if True:
    evalMain(['--path', 'save_dir/redo4',
            '--env_group', '4',
            '--make_vids',
            '--make_evals', '--episodes', '20'])
            
print('visualize')
            
visualizeMain(['--path', 'save_dir/redo4'])
