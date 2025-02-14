
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
    'sigmoid_temp': 1.0,
    'combiner': False,
    'num_beliefs': 1,
    'vision_prob': 1.0
}

BASE_RANDOM = {k: BASE_CONFIG[k] for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision']}

TEMP_CONFIGS = {
    f'a-hardcoded-t{temp}': {'sigmoid_temp': temp, 'vision_prob': 1.0, 'num_beliefs': 1}
    for temp in [1, 2, 5, 8, 10, 15, 20, 30, 50, 100, 200]
}



BASE_NEURAL_CONFIGS = {
    'a-hardcoded': {},
    'a-neural-split': {k: True for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision']},
    'a-neural-shared': {k: True for k in BASE_CONFIG},
    'a-mix-n-perception': {'perception': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-belief-shared': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-belief-split': {'my_belief': True, 'op_belief': True, 'shared_belief': False, 'shared_decision': True},
    'a-mix-n-belief-combiner-shared': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'combiner': True},
    'a-mix-n-belief-combiner-split': {'my_belief': True, 'op_belief': True, 'shared_belief': False, 'shared_decision': True, 'combiner': True},
    'a-mix-n-belief-op': {'op_belief': True, 'shared_decision': True},
    'a-mix-n-belief-my': {'my_belief': True, 'shared_decision': True},
    'a-mix-n-combiner': {'combiner': True, 'shared_decision': True},
    'a-mix-n-decision-shared': {'my_greedy_decision': True, 'op_greedy_decision': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-decision-op': {'op_greedy_decision': True, 'shared_belief': True},
    'a-mix-n-decision-my': {'my_greedy_decision': True, 'shared_belief': True},
    'a-mix-n-output': {'final_output': True, 'shared_belief': True, 'shared_decision': True}
}

NEURAL_CONFIGS = {}
for base_name, base_config in BASE_NEURAL_CONFIGS.items():
    NEURAL_CONFIGS[base_name] = {**base_config, 'vision_prob': 1.0, 'num_beliefs': 1}
    for vision_prob in [0.75, 0.5]:
        for num_beliefs in [1, 3, 5]:
            name = f"{base_name}-v{int(vision_prob * 100)}-b{num_beliefs}"
            NEURAL_CONFIGS[name] = {
                **base_config,
                'vision_prob': vision_prob,
                'num_beliefs': num_beliefs
            }

#print(NEURAL_CONFIGS.keys())

RANDOM_VARIANTS = {}
for model_name, config in NEURAL_CONFIGS.items():
    # add random variants
    if 'mixed' in model_name and 'shared' not in model_name:
        neural_key = next(k for k, v in config.items() if v is True and k not in ['shared_belief', 'shared_decision'])
        base_config = {k: v for k, v in config.items() if k in ['shared_belief', 'shared_decision']}
        for prob in [100, 50]:
            variant_name = model_name.replace('mix-n-', 'mix-r-') + f"-{prob}"
            RANDOM_VARIANTS[variant_name] = (base_config, {neural_key: prob/100})

MODEL_SPECS = {name: (config, {}) for name, config in {**NEURAL_CONFIGS, **TEMP_CONFIGS}.items()}
MODEL_SPECS.update(RANDOM_VARIANTS)