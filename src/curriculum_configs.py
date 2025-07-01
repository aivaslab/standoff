class CurriculumConfig:
    def __init__(self):
        self.curriculum_stages = [
            {
                'stage_name': 'solo',
                'batches': 500,
                'data_regimes': ['s1'],
                'trainable_modules': [
                    'treat_perception_my', 'vision_perception_my', 'presence_perception_my',
                    'my_belief', 'my_decision', 'my_combiner'
                ],
                'neural_modules': [
                    'my_treat', 'vision_my', 'presence_my', 
                    'my_belief', 'my_decision', 'combiner'
                ],
                'frozen_modules': [
                    'treat_perception_op', 'vision_perception_op', 'presence_perception_op',
                    'op_belief', 'op_decision', 'op_combiner'
                ],
                'copy_weights': None
            },
            {
                'stage_name': 'informed',
                'batches': 500,
                'data_regimes': ['s2'],
                'trainable_modules': ['treat_perception_op'],
                'neural_modules': [
                    'my_treat', 'vision_my', 'presence_my', 'my_belief', 'my_decision', 'combiner',
                    'op_treat'
                ],
                'frozen_modules': [
                    'treat_perception_my', 'vision_perception_my', 'presence_perception_my',
                    'my_belief', 'my_decision', 'my_combiner',
                    'vision_perception_op', 'presence_perception_op', 'op_belief', 'op_decision', 'op_combiner'
                ],
                'copy_weights': {
                    'treat_perception_my': 'treat_perception_op',
                    'vision_perception_my': 'vision_perception_op', 
                    'presence_perception_my': 'presence_perception_op'
                }
            },
            {
                'stage_name': 'tom_simple',
                'batches': 500,
                'data_regimes': ['s21'],
                'trainable_modules': ['op_belief', 'op_combiner'],
                'neural_modules': [
                    'my_treat', 'vision_my', 'presence_my', 'my_belief', 'my_decision', 'combiner',
                    'op_treat', 'vision_op', 'presence_op', 'op_belief'
                ],
                'frozen_modules': [
                    'treat_perception_my', 'vision_perception_my', 'presence_perception_my',
                    'my_belief', 'my_decision', 'my_combiner',
                    'treat_perception_op', 'vision_perception_op', 'presence_perception_op',
                    'op_decision'
                ],
                'copy_weights': None
            }
        ]

def copy_module_weights(source_module, target_module):
    if hasattr(source_module, 'neural_network') and hasattr(target_module, 'neural_network'):
        if source_module.neural_network is not None and target_module.neural_network is not None:
            target_module.neural_network.load_state_dict(source_module.neural_network.state_dict())
            return
    
    source_params = dict(source_module.named_parameters())
    target_params = dict(target_module.named_parameters())
    
    for name, source_param in source_params.items():
        if name in target_params:
            target_params[name].data.copy_(source_param.data)

def apply_curriculum_stage(model, stage_config, train_sets_dict):
    all_modules = {
        'treat_perception_my': model.treat_perception_my,
        'treat_perception_op': model.treat_perception_op,
        'vision_perception_my': model.vision_perception_my,
        'vision_perception_op': model.vision_perception_op,
        'presence_perception_my': model.presence_perception_my,
        'presence_perception_op': model.presence_perception_op,
        'my_belief': model.my_belief,
        'op_belief': model.op_belief,
        'my_decision': model.my_decision,
        'op_decision': model.op_decision,
        'my_combiner': model.my_combiner,
        'op_combiner': model.op_combiner
    }
    
    if stage_config['copy_weights']:
        for source_name, target_name in stage_config['copy_weights'].items():
            if source_name in all_modules and target_name in all_modules:
                copy_module_weights(all_modules[source_name], all_modules[target_name])
    
    for module_name in all_modules:
        if module_name in stage_config['frozen_modules']:
            freeze_module_parameters(all_modules[module_name])
        elif module_name in stage_config['trainable_modules']:
            unfreeze_module_parameters(all_modules[module_name])
        else:
            freeze_module_parameters(all_modules[module_name])
    
    stage_train_sets = []
    for regime in stage_config['data_regimes']:
        if regime in train_sets_dict:
            stage_train_sets.extend(train_sets_dict[regime])
    
    return stage_train_sets

def freeze_module_parameters(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module_parameters(module):
    for param in module.parameters():
        param.requires_grad = True