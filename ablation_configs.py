BASE_CONFIG = {
    'my_treat': False,
    'op_treat': False,
    'vision_my': False,
    'vision_op': False,
    'presence_my': False,
    'presence_op': False,
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
    'end2end': False,
    'opponent_perception': False,
    'output_type': 'my_decision',
    'arch': 'mlp',
    'archx': 'mlp',
    'size_swap': False,
    'use_oracle': False,
    'pad': False,
    'split': False,
    'shared': False,
    'full': False,
    'full_infer_decisions': False,
    'sim_type': "r",
    'use_sim_loss': True,
    'use_gt_sim': False,
    'use_behavior_loss': False,
    'use_self_belief_loss': False,
}

BASE_RANDOM = {k: BASE_CONFIG[k] for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision']}

TEMP_CONFIGS = {
    f'a-hardcoded-t{temp}': {'sigmoid_temp': temp, 'vision_prob': 1.0, 'num_beliefs': 1}
    for temp in [1, 2, 5, 8, 10, 15, 20, 30, 50, 100, 200]
}

BASE_NEURAL_CONFIGS = {
    'a-hardcoded': {'shared_decision': True},

    'a-neural-split': {k: True for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision', 'shared_treat', 'end2end', 'opponent_perception', 'output_type', 'arch']},
    'a-neural-split-transformer': {k: True for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision', 'shared_treat', 'end2end', 'opponent_perception', 'output_type', 'arch']},
    'a-neural-belief-shared': {k: True for k in BASE_CONFIG if k not in ['shared_decision', 'shared_treat', 'end2end', 'opponent_perception', 'output_type', 'arch']},
    'a-neural-decision-shared': {k: True for k in BASE_CONFIG if k not in ['shared_belief', 'shared_treat', 'end2end', 'opponent_perception', 'output_type', 'arch']},
    'a-neural-treat-shared': {k: True for k in BASE_CONFIG if k not in ['shared_belief', 'shared_decision', 'end2end', 'opponent_perception', 'output_type', 'arch']},
    'a-neural-shared': {k: True for k in BASE_CONFIG},
    'a-neural-shared-transformer': {k: True for k in BASE_CONFIG},
    'a-neural-detach': {k: True for k in BASE_CONFIG},

    'a-mix-n-vision-op': {'vision_op': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-presence-op': {'presence_op': True, 'shared_belief': True, 'shared_decision': True},

    'a-mix-n-treat-my': {'my_treat': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-treat-op': {'op_treat': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-treat-op-transformer': {'op_treat': True, 'shared_belief': True, 'shared_decision': True, 'archx': 'transformer'},
    'a-mix-n-treat-split': {'my_treat': True, 'op_treat': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-treat-split-transformer': {'my_treat': True, 'op_treat': True, 'shared_belief': True, 'shared_decision': True, 'archx': 'transformer'},
    'a-mix-n-treat-shared': {'my_treat': True, 'op_treat': True, 'shared_belief': True, 'shared_decision': True, 'shared_treat': True, 'detach': False},
    'a-mix-n-treat-shared-transformer': {'my_treat': True, 'op_treat': True, 'shared_belief': True, 'shared_decision': True, 'shared_treat': True, 'detach': False, 'archx': 'transformer'},
    'a-mix-n-treat-detach': {'my_treat': True, 'op_treat': True, 'shared_belief': True, 'shared_decision': True, 'shared_treat': True, 'detach': True},

    'a-mix-n-perception-my': {'my_treat': True, 'vision_my': True, 'presence_my': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-perception-op': {'op_treat': True,  'vision_op': True, 'presence_op': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-perception-split': {'my_treat': True, 'vision_my': True, 'presence_my': True, 'vision_op': True, 'presence_op': True,'op_treat': True, 'shared_belief': True, 'shared_decision': True},
    'a-mix-n-perception-shared': {'my_treat': True, 'vision_my': True, 'presence_my': True, 'vision_op': True, 'presence_op': True,'op_treat': True, 'shared_belief': True, 'shared_decision': True, 'shared_treat': True, 'detach': False},
    'a-mix-n-perception-detach': {'my_treat': True, 'vision_my': True, 'presence_my': True, 'vision_op': True, 'presence_op': True,'op_treat': True, 'shared_belief': True, 'shared_decision': True, 'shared_treat': True, 'detach': True},

    'a-mix-n-belief-op': {'op_belief': True, 'shared_decision': True, 'shared_treat': True},
    'a-mix-n-belief-op-transformer': {'op_belief': True, 'shared_decision': True, 'shared_treat': True, 'archx': 'transformer'},
    'a-mix-n-belief-my': {'my_belief': True, 'shared_decision': True, 'shared_treat': True},
    'a-mix-n-belief-split': {'my_belief': True, 'op_belief': True, 'shared_decision': True, 'shared_treat': True},
    'a-mix-n-belief-split-transformer': {'my_belief': True, 'op_belief': True, 'shared_decision': True, 'shared_treat': True, 'archx': 'transformer'},
    'a-mix-n-belief-shared': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'detach': False, 'shared_treat': True},
    'a-mix-n-belief-shared-transformer': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'detach': False, 'shared_treat': True, 'archx': 'transformer'},
    'a-mix-n-belief-detach': {'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'detach': True, 'shared_treat': True},

    'a-mix-n-combiner-split': {'combiner': True, 'shared_decision': True, 'shared_treat': True},
    'a-mix-n-combiner-shared': {'combiner': True, 'shared_decision': True, 'shared_combiner': True, 'detach': False, 'shared_treat': True},
    'a-mix-n-combiner-detach': {'combiner': True, 'shared_decision': True, 'shared_combiner': True, 'detach': True, 'shared_treat': True},

    'a-mix-n-decision-op': {'op_decision': True, 'shared_belief': True, 'shared_treat': True},
    'a-mix-n-decision-op-transformer': {'op_decision': True, 'shared_belief': True, 'shared_treat': True, 'archx': 'transformer'},
    'a-mix-n-decision-my': {'my_decision': True, 'shared_belief': True, 'shared_treat': True},
    'a-mix-n-decision-split': {'my_decision': True, 'op_decision': True, 'shared_belief': True, 'shared_treat': True},
    'a-mix-n-decision-split-transformer': {'my_decision': True, 'op_decision': True, 'shared_belief': True, 'shared_treat': True, 'archx': 'transformer'},
    'a-mix-n-decision-shared': {'my_decision': True, 'op_decision': True, 'shared_belief': True, 'shared_decision': True, 'detach': False, 'shared_treat': True},
    'a-mix-n-decision-shared-transformer': {'my_decision': True, 'op_decision': True, 'shared_belief': True, 'shared_decision': True, 'detach': False, 'shared_treat': True, 'archx': 'transformer'},
    'a-mix-n-decision-detach': {'my_decision': True, 'op_decision': True, 'shared_belief': True, 'shared_decision': True, 'detach': True, 'shared_treat': True},

    'a-mix-n-all-my': {'my_decision': True, 'my_belief': True, 'my_treat': True, 'vision_my': True, 'presence_my': True,},
    'a-mix-n-all-op': {'op_decision': True, 'op_belief': True, 'op_treat': True, 'vision_op': True, 'presence_op': True},
    'a-mix-n-all-op-transformer': {'op_decision': True, 'op_belief': True, 'op_treat': True, 'vision_op': True, 'presence_op': True},
    'a-mix-n-all-shared': {'my_treat': True, 'op_treat': True, 'vision_my': True, 'presence_my': True, 'vision_op': True, 'presence_op': True, 'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'detach': False, 'shared_treat': True},
    'a-mix-n-all-shared-transformer': {'my_treat': True, 'op_treat': True, 'vision_my': True, 'presence_my': True, 'vision_op': True, 'presence_op': True, 'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'detach': False, 'shared_treat': True},
    'a-mix-n-all-detach': {'my_treat': True, 'op_treat': True, 'vision_my': True, 'presence_my': True, 'vision_op': True, 'presence_op': True, 'my_belief': True, 'op_belief': True, 'shared_belief': True, 'shared_decision': True, 'detach': True, 'shared_treat': True},
    'a-mix-n-all-split': {'my_treat': True, 'op_treat': True, 'vision_my': True, 'presence_my': True, 'vision_op': True, 'presence_op': True, 'my_belief': True, 'op_belief': True, 'shared_belief': False, 'shared_decision': False, 'detach': False},
    'a-mix-n-all-split-transformer': {'my_treat': True, 'op_treat': True, 'vision_my': True, 'presence_my': True, 'vision_op': True, 'presence_op': True, 'my_belief': True, 'op_belief': True, 'shared_belief': False, 'shared_decision': False, 'detach': False},
}

BASE_NEURAL_CONFIGS['a-neural-split']['detach'] = False
BASE_NEURAL_CONFIGS['a-neural-detach']['detach'] = True
BASE_NEURAL_CONFIGS['a-neural-shared']['detach'] = False
BASE_NEURAL_CONFIGS['a-neural-belief-shared']['detach'] = False
BASE_NEURAL_CONFIGS['a-neural-decision-shared']['detach'] = False
BASE_NEURAL_CONFIGS['a-neural-treat-shared']['detach'] = False

for x in ['mlp', 'transformer32', 'transformer128']:
    for y in ['', '-pad']:
        for sim_type in ["", "-r", "-i", "-ri", "-rp"]:
            for sim_loss in ["", "-nsl"]:
                for gt_sim in ["", "-gts"]:
                    for decision_loss in ["", "-dl"]:
                        for output_type in ["op_decision", "both_decision", "both_belief", "belief"]:
                            output_name = output_type if output_type != "op_decision" else ""
                            BASE_NEURAL_CONFIGS[f'a-simv2-single-{x}{y}{sim_type}{sim_loss}{gt_sim}{decision_loss}{output_name}'] = {'arch': x, 'pad': y != '', "sim_type": sim_type, "output_type": 'op_decision', 'use_sim_loss': sim_loss=="", 'use_gt_sim': gt_sim != '', 'use_behavior_loss': decision_loss != ''}
                            BASE_NEURAL_CONFIGS[f'a-simv2-split-{x}{y}{sim_type}{sim_loss}{gt_sim}{decision_loss}{output_name}'] = {'arch': x, 'pad': y != '', "sim_type": sim_type, "output_type": 'op_decision', "split": True, 'use_sim_loss': sim_loss=="", 'use_gt_sim': gt_sim != '', 'use_behavior_loss': decision_loss != ''}
                            BASE_NEURAL_CONFIGS[f'a-simv2-shared-{x}{y}{sim_type}{sim_loss}{gt_sim}{decision_loss}{output_name}'] = {'arch': x, 'pad': y != '', "sim_type": sim_type, "output_type": 'op_decision', "shared": True, 'use_sim_loss': sim_loss=="", 'use_gt_sim': gt_sim != '', 'use_behavior_loss': decision_loss != ''}


        BASE_NEURAL_CONFIGS[f'a-opbelief-{x}{y}'] = {'end2end': True, 'opponent_perception': False, 'output_type': 'op_belief', 'arch': x, 'pad': y != ''}
        BASE_NEURAL_CONFIGS[f'a-opdecision-{x}{y}'] = {'end2end': True, 'opponent_perception': False, 'output_type': 'op_decision', 'arch': x, 'pad': y != ''}
        BASE_NEURAL_CONFIGS[f'a-opbelief-sym-{x}{y}'] = {'end2end': True, 'opponent_perception': True, 'output_type': 'op_belief', 'arch': x, 'pad': y != ''}
        BASE_NEURAL_CONFIGS[f'a-opdecision-sym-{x}{y}'] = {'end2end': True, 'opponent_perception': True, 'output_type': 'op_decision', 'arch': x, 'pad': y != ''}

        BASE_NEURAL_CONFIGS[f'a-opbeliefM-{x}{y}'] = {'end2end': True, 'opponent_perception': False, 'output_type': 'multi', 'arch': x, 'pad': y != ''}
        BASE_NEURAL_CONFIGS[f'a-opdecisionM-{x}{y}'] = {'end2end': True, 'opponent_perception': False, 'output_type': 'multi', 'arch': x, 'pad': y != ''}
        BASE_NEURAL_CONFIGS[f'a-opbeliefM-sym-{x}{y}'] = {'end2end': True, 'opponent_perception': True, 'output_type': 'multi', 'arch': x, 'pad': y != ''}
        BASE_NEURAL_CONFIGS[f'a-opdecisionM-sym-{x}{y}'] = {'end2end': True, 'opponent_perception': True, 'output_type': 'multi', 'arch': x, 'pad': y != ''}

        for z, size_swap in zip(['swap',''], [True, False]):
            BASE_NEURAL_CONFIGS[f'a-split-{x}{y}-{z}'] = {'end2end': True, 'opponent_perception': False, 'output_type': 'multi', 'arch': x, 'size_swap': size_swap, 'pad': y != '', 'split': True, 'full_infer_decisions': True}
            BASE_NEURAL_CONFIGS[f'a-shared-{x}{y}-{z}'] = {'end2end': True, 'opponent_perception': False, 'output_type': 'multi', 'arch': x, 'size_swap': size_swap, 'pad': y != '', 'shared': True, 'full_infer_decisions': True}

            BASE_NEURAL_CONFIGS[f'a-split-sym-{x}{y}-{z}'] = {'end2end': True, 'opponent_perception': True, 'output_type': 'multi', 'arch': x, 'size_swap': size_swap, 'pad': y != '', 'split': True, 'full_infer_decisions': True}
            BASE_NEURAL_CONFIGS[f'a-shared-sym-{x}{y}-{z}'] = {'end2end': True, 'opponent_perception': True, 'output_type': 'multi', 'arch': x, 'size_swap': size_swap, 'pad': y != '', 'shared': True, 'full_infer_decisions': True}

            for o, use_oracle in zip(['oracle',''], [True, False]):
                BASE_NEURAL_CONFIGS[f'a-fullM-{x}{y}-{z}-{o}'] = {'end2end': True, 'opponent_perception': False, 'output_type': 'multi', 'arch': x, 'size_swap': size_swap, 'use_oracle': use_oracle, 'pad': y != '', 'full': True, 'full_infer_decisions': True}
                BASE_NEURAL_CONFIGS[f'a-fullM-sym-{x}{y}-{z}-{o}'] = {'end2end': True, 'opponent_perception': True, 'output_type': 'multi', 'arch': x, 'size_swap': size_swap, 'use_oracle': use_oracle, 'pad': y != '', 'full': True, 'full_infer_decisions': True}

NEURAL_CONFIGS = {}
for base_name, base_config in BASE_NEURAL_CONFIGS.items():
    NEURAL_CONFIGS[base_name] = {**base_config, 'vision_prob': 1.0, 'num_beliefs': 1}
    if "transformer" in base_name:
        NEURAL_CONFIGS[base_name]['archx'] = 'transformer'
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
            if "transformer" in base_name:
                NEURAL_CONFIGS[name]['archx'] = 'transformer'

RANDOM_VARIANTS = {}
for model_name, config in NEURAL_CONFIGS.items():
    if 'mix' in model_name and 'shared' not in model_name and '0' not in model_name and '5' not in model_name:
        neural_key = next(k for k, v in config.items() if v is True and k not in ['shared_belief', 'shared_decision'])
        base_config = {k: False for k, _ in config.items() if k in ['shared_belief', 'shared_decision']}
        base_config[neural_key] = False
        for prob in [100, 50]:
            variant_name = model_name.replace('mix-n-', 'mix-r-') + f"-{prob}"
            RANDOM_VARIANTS[variant_name] = (base_config, {neural_key: prob/100})

MODEL_SPECS = {name: (config, {}) for name, config in {**NEURAL_CONFIGS, **TEMP_CONFIGS}.items()}
MODEL_SPECS.update(RANDOM_VARIANTS)