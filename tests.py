from src.train import main as trainMain
from src.evaluate import main as evalMain
from src.visualize import main as visualizeMain
import sys
import os 

sys.path.append(os.getcwd())

print('train')

trainMain(['--experiment_name', 'x',
            '--env_group', '1',
            '--timesteps', '100',
            '--checkpoints', '1',
            '--overwrite', 'True', 
            '--savePath', 'save_dir'])
            
print('evaluate')
            
evalMain(['--path', 'save_dir/x',
            '--env_group', '1',
            '--episodes', '2',])
            
print('visualize')
            
visualizeMain(['--path', 'save_dir/x'])
