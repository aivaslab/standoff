
BASE_CONFIG = {
    'perception': False,
    'my_belief': False,
    'op_belief': False,
    'my_greedy_decision': False,
    'op_greedy_decision': False,
    'sub_decision': False,
    'final_output': False,
    'shared_belief': False,
    'shared_decision': False,
    'sigmoid_temp': 20.0
}

BASE_RANDOM = {k: 0.0 for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision']}

TEMP_CONFIGS = {
    'a-hardcoded-t1': {'sigmoid_temp': 1},
    'a-hardcoded-t2': {'sigmoid_temp': 2},
    'a-hardcoded-t5': {'sigmoid_temp': 5},
    'a-hardcoded-t8': {'sigmoid_temp': 8},
    'a-hardcoded-t10': {'sigmoid_temp': 10},
    'a-hardcoded-t15': {'sigmoid_temp': 15},
    'a-hardcoded-t20': {'sigmoid_temp': 20},
    'a-hardcoded-t30': {'sigmoid_temp': 30},
    'a-hardcoded-t50': {'sigmoid_temp': 50},
    'a-hardcoded-t100': {'sigmoid_temp': 100},
    'a-hardcoded-t200': {'sigmoid_temp': 200},
}

NEURAL_CONFIGS = {
    'a-hardcoded': {},
    'a-neural-split': {k: True for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision']},
    'a-neural-shared': {k: True for k in BASE_CONFIG},
    'a-mixed-n-perception': {'perception': True, 'shared_belief': True, 'shared_decision': True},
    'a-mixed-n-belief-shared': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True},
    'a-mixed-n-belief-op': {'op_belief': True, 'shared_decision': True},
    'a-mixed-n-belief-my': {'my_belief': True, 'shared_decision': True},
    'a-mixed-n-decision-shared': {'my_greedy_decision': True, 'op_greedy_decision': True, 'shared_belief': True, 'shared_decision': True},
    'a-mixed-n-decision-op': {'op_greedy_decision': True, 'shared_belief': True},
    'a-mixed-n-decision-my': {'my_greedy_decision': True, 'shared_belief': True},
    'a-mixed-n-output': {'final_output': True, 'shared_belief': True, 'shared_decision': True}
}

RANDOM_VARIANTS = {}
for model_name, config in NEURAL_CONFIGS.items():
    if 'mixed' in model_name and 'shared' not in model_name:
        neural_key = next(k for k, v in config.items() if v is True and k not in ['shared_belief', 'shared_decision'])
        base_config = {k: v for k, v in config.items() if k in ['shared_belief', 'shared_decision']}
        for prob in [100, 50]:
            variant_name = model_name.replace('mixed-n-', 'mixed-r-') + f"-{prob}"
            RANDOM_VARIANTS[variant_name] = (base_config, {neural_key: prob/100})

MODEL_SPECS = {name: (config, {}) for name, config in {**NEURAL_CONFIGS, **TEMP_CONFIGS}.items()}
MODEL_SPECS.update(RANDOM_VARIANTS)