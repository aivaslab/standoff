
BASE_CONFIG = {
    'my_treat': False,
    'op_treat': False,
    'vision': False,
    'presence': False,
    'my_belief': False,
    'op_belief': False,
    'my_decision': False,
    'op_decision': False,
    'shared_treat': False,
    'shared_belief': False,
    'shared_decision': False,
    'shared_combiner': False,
    'sigmoid_temp': 90.0,
    'combiner': False,
    'use_combiner': False,
    'num_beliefs': 1,
    'vision_prob': 1.0,
    'mix_neurons': False,
}

BASE_RANDOM = {k: BASE_CONFIG[k] for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision']}

TEMP_CONFIGS = {
    f'a-hardcoded-t{temp}': {'sigmoid_temp': temp, 'vision_prob': 1.0, 'num_beliefs': 1}
    for temp in [1, 2, 5, 8, 10, 15, 20, 30, 50, 100, 200]
}



BASE_NEURAL_CONFIGS = {
    'a-hardcoded': {'shared_decision': True},

    'a-neural-split': {k: True for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision', 'shared_treat']},
    'a-neural-belief-shared': {k: True for k in BASE_CONFIG if k not in ['shared_decision', 'shared_treat']},
    'a-neural-decision-shared': {k: True for k in BASE_CONFIG if k not in ['shared_belief', 'shared_treat']},
    'a-neural-treat-shared': {k: True for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision']},
    'a-neural-shared': {k: True for k in BASE_CONFIG},
    'a-neural-detach': {k: True for k in BASE_CONFIG},

    'a-mix-n-vision-op': {'vision': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-presence-op': {'presence': True, 'shared_belief': True, 'shared_decision': True},

    'a-mix-n-treat-my': {'my_treat': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-treat-op': {'op_treat': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-treat-split': {'my_treat': True, 'op_treat': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-treat-shared': {'my_treat': True, 'op_treat': True, 'shared_belief': True, 'shared_decision': True, 'shared_treat': True},
    'a-mix-n-treat-detach': {'my_treat': True, 'op_treat': True, 'shared_belief': True, 'shared_decision': True, 'shared_treat': True, 'detach': True},

    'a-mix-n-belief-op': {'op_belief': True, 'shared_decision': True},
    'a-mix-n-belief-my': {'my_belief': True, 'shared_decision': True},
    'a-mix-n-belief-split': {'my_belief': True, 'op_belief': True, 'shared_decision': True},
    'a-mix-n-belief-shared': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-belief-detach': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'detach': True},

    'a-mix-n-combiner': {'combiner': True, 'shared_decision': True},

    'a-mix-n-decision-op': {'op_decision': True, 'shared_belief': True},
    'a-mix-n-decision-my': {'my_decision': True, 'shared_belief': True},
    'a-mix-n-decision-split': {'my_decision': True, 'op_decision': True, 'shared_belief': True,},
    'a-mix-n-decision-shared': {'my_decision': True, 'op_decision': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-decision-detach': {'my_decision': True, 'op_decision': True, 'shared_belief': True, 'shared_decision': True, 'detach': True},

    'a-mix-n-all-my': {'my_decision': True, 'my_belief': True, 'my_treat': True},
    'a-mix-n-all-op': {'op_decision': True, 'op_belief': True, 'op_treat': True},
    'a-mix-n-all-shared': {'my_treat': True, 'op_treat': True, 'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-all-detach': {'my_treat': True, 'op_treat': True, 'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'detach': True},
    'a-mix-n-all-split': {'my_treat': True, 'op_treat': True, 'my_belief': True, 'op_belief': True, 'shared_belief': False, 'shared_decision': False},

    #'a-mix-n-belief-combiner-shared': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'combiner': True, 'shared_combiner': True},
    #'a-mix-n-belief-combiner-split': {'my_belief': True, 'op_belief': True, 'shared_belief': False, 'shared_decision': False, 'combiner': True, 'shared_combiner': False},
    #'a-mix-n-belief-comb-decision-shared': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'combiner': True, 'decision': True, 'shared_combiner': True},
    #'a-mix-n-belief-comb-decision-split': {'my_belief': True, 'op_belief': True, 'shared_belief': False, 'shared_decision': False, 'combiner': True, 'decision': True, 'shared_combiner': False},
}

BASE_NEURAL_CONFIGS['a-neural-detach']['detach'] = True

NEURAL_CONFIGS = {}
for base_name, base_config in BASE_NEURAL_CONFIGS.items():
    NEURAL_CONFIGS[base_name] = {**base_config, 'vision_prob': 1.0, 'num_beliefs': 1}
    for vision_prob in [0.75, 0.5]:
        for num_beliefs in [1, 3, 5]:
            name = f"{base_name}-v{int(vision_prob * 100)}-b{num_beliefs}"
            NEURAL_CONFIGS[name] = {
                **base_config,
                'vision_prob': vision_prob,
                'num_beliefs': num_beliefs,
                'sigmoid_temp': 90,
                'use_combiner': False if num_beliefs == 1 else True
            }


RANDOM_VARIANTS = {}
for model_name, config in NEURAL_CONFIGS.items():
    # add random variants
    if 'mix' in model_name and 'shared' not in model_name and '0' not in model_name and '5' not in model_name:
        neural_key = next(k for k, v in config.items() if v is True and k not in ['shared_belief', 'shared_decision'])
        base_config = {k: False for k, _ in config.items() if k in ['shared_belief', 'shared_decision']} # no sharing in random modules!!!!!
        base_config[neural_key] = False
        for prob in [100, 50]:
            variant_name = model_name.replace('mix-n-', 'mix-r-') + f"-{prob}"
            RANDOM_VARIANTS[variant_name] = (base_config, {neural_key: prob/100})


MODEL_SPECS = {name: (config, {}) for name, config in {**NEURAL_CONFIGS, **TEMP_CONFIGS}.items()}
MODEL_SPECS.update(RANDOM_VARIANTS)