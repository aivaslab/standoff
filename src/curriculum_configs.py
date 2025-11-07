def extract_regime_and_base(name):
    regime = 's1'
    base = name
    for r in ['s21', 's3', 's2', 's1', 's22']:
        if name.endswith('_' + r):
            regime = r
            base = name[:-len('_' + r)]
            break
    return regime, base

class CurriculumConfig:
    def __init__(self, curriculum_name):

        module_map = {
                'perception_my': ['treat_perception_my', 'vision_perception_my', 'presence_perception_my'],
                'perception_op': ['treat_perception_op', 'vision_perception_op', 'presence_perception_op'],
                'perception_both': ['treat_perception_my', 'vision_perception_my', 'presence_perception_my', 'treat_perception_op', 'vision_perception_op', 'presence_perception_op'],
                'belief_my': ['my_belief'],
                'belief_op': ['op_belief', 'op_belief_t'],
                'belief_both': ['my_belief', 'op_belief'],
                'decision_my': ['my_decision'],
                'decision_op': ['op_decision'],
                'decision_both': ['my_decision', 'op_decision'],
            }

        all_modules = [
                        'treat_perception_my', 
                        'vision_perception_my', 
                        'presence_perception_my',
                        'treat_perception_op', 
                        'vision_perception_op', 
                        'presence_perception_op', 
                        'my_belief',
                        'op_belief',
                        'my_decision',
                        'op_decision', 
                        'end2end_model'
                    ]

        print('cur got', curriculum_name)

        if curriculum_name == "three":
            self.curriculum_stages = [
                {
                    'stage_name': 'solo',
                    'batches': 500,
                    'data_regimes': ['s1'],
                    'trainable_modules': [
                        'treat_perception_my', 
                        'vision_perception_my', 
                        'presence_perception_my',
                        'my_belief',
                        'my_decision',
                    ],
                    'copy_weights': None,
                },
                {
                    'stage_name': 'informed',
                    'batches': 750,
                    'data_regimes': ['s2'],
                    'trainable_modules': [
                        'treat_perception_op', 
                        'presence_perception_op', 
                        'my_decision'],
                    'copy_weights': {
                        'treat_perception_my': 'treat_perception_op',
                        'vision_perception_my': 'vision_perception_op', 
                        'presence_perception_my': 'presence_perception_op',
                        'my_belief': 'op_belief',
                        'my_decision': 'op_decision'
                    },
                },
                {
                    'stage_name': 'tom_simple',
                    'batches': 1000,
                    'data_regimes': ['s21'],
                    'trainable_modules': [
                        'op_belief', 
                        'vision_perception_op', 
                        'op_decision', 
                        'my_decision'],
                    'copy_weights': None,
                }
            ]
        elif "everything" in curriculum_name:
            regime, base_curriculum_name = extract_regime_and_base(curriculum_name)
            self.curriculum_stages = [
                {
                    'stage_name': curriculum_name,
                    'batches': 5000,
                    'data_regimes': [regime],
                    'trainable_modules': all_modules,
                    'copy_weights': None,
                },
            ]
        elif "end2end" in curriculum_name:
            regime, base_curriculum_name = extract_regime_and_base(curriculum_name)
            print('regime', regime)
            self.curriculum_stages = [
                {
                    'stage_name': base_curriculum_name,
                    'batches': 20000,
                    'data_regimes': [regime],
                    'trainable_modules': ['end2end_model'], #['end2end_model']
                    'copy_weights': None,
                },
            ]
        else:

            regime, base_curriculum_name = extract_regime_and_base(curriculum_name)
            
            if base_curriculum_name in module_map:
                self.curriculum_stages = [{
                    'stage_name': base_curriculum_name,
                    'batches': 10000,
                    'data_regimes': [regime],
                    'trainable_modules': module_map[base_curriculum_name],
                    'copy_weights': None,
                }]

            elif base_curriculum_name.endswith('_then_all'):
                temp_name = base_curriculum_name[:-9]
                if temp_name in module_map:
                    self.curriculum_stages = [
                        {
                            'stage_name': temp_name,
                            'batches': 2000,
                            'data_regimes': [regime],
                            'trainable_modules': module_map[temp_name],
                            'copy_weights': None,
                        },
                        {
                            'stage_name': 'all',
                            'batches': 2000,
                            'data_regimes': [regime],
                            'trainable_modules': all_modules,
                            'copy_weights': None,
                        }
                    ]

            elif base_curriculum_name.endswith('_then_else'):
                temp_name = base_curriculum_name[:-10]
                if temp_name in module_map:
                    self.curriculum_stages = [
                        {
                            'stage_name': temp_name,
                            'batches': 2000,
                            'data_regimes': [regime],
                            'trainable_modules': module_map[temp_name],
                            'copy_weights': None,
                        },
                        {
                            'stage_name': 'all',
                            'batches': 2000,
                            'data_regimes': [regime],
                            'trainable_modules': [module for module in all_modules if module not in module_map[temp_name]],
                            'copy_weights': None,
                        }
                    ]

            elif base_curriculum_name.startswith('else_then_'):
                temp_name = base_curriculum_name[10:]
                if temp_name in module_map:
                    self.curriculum_stages = [
                        {
                            'stage_name': 'all',
                            'batches': 2000,
                            'data_regimes': [regime],
                            'trainable_modules': [module for module in all_modules if module not in module_map[temp_name]],
                            'copy_weights': None,
                        },
                        {
                            'stage_name': temp_name,
                            'batches': 2000,
                            'data_regimes': [regime],
                            'trainable_modules': module_map[temp_name],
                            'copy_weights': None,
                        }
                    ]

            elif base_curriculum_name.startswith('all_then_'):
                temp_name = base_curriculum_name[9:]
                if temp_name in module_map:
                    self.curriculum_stages = [
                        {
                            'stage_name': 'all',
                            'batches': 2000,
                            'data_regimes': [regime],
                            'trainable_modules': all_modules,
                            'copy_weights': None,
                        },
                        {
                            'stage_name': temp_name,
                            'batches': 2000,
                            'data_regimes': [regime],
                            'trainable_modules': module_map[temp_name],
                            'copy_weights': None,
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
        'op_combiner': model.op_combiner,
        'end2end_model': model.end2end_model
    }
    
    if not hasattr(model, '_ever_trained'):
        model._ever_trained = set()
    copied_to_modules = set()
    
    if stage_config['copy_weights']:
        for source_name, target_name in stage_config['copy_weights'].items():
            if source_name in all_modules and target_name in all_modules:
                copy_module_weights(all_modules[source_name], all_modules[target_name])
                copied_to_modules.add(target_name)
    
    for module_name, module in all_modules.items():
        if module is None:
            continue
        if module_name in stage_config['trainable_modules']:
            if hasattr(module, 'use_neural'):
                module.use_neural = True
            unfreeze_module_parameters(module)
            model._ever_trained.add(module_name)
        elif module_name in model._ever_trained or module_name in copied_to_modules:
            if hasattr(module, 'use_neural'):
                module.use_neural = True
            freeze_module_parameters(module)
        else:
            if hasattr(module, 'use_neural'):
                module.use_neural = False
            freeze_module_parameters(module)
    
    trainable = [
        name for name, module in model.get_module_dict().items()
        if module is not None and any(p.requires_grad for p in module.parameters())
    ]
    print(f"Trainable: {trainable}")
    
    

def freeze_module_parameters(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module_parameters(module):
    for param in module.parameters():
        param.requires_grad = True