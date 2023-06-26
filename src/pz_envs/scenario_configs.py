from itertools import product


def parameter_generator(space, params={}):
    if space:
        key, value = next(iter(space.items()))
        if callable(value):
            values = value(params)
        else:
            values = value
        for v in values:
            new_params = params.copy()
            new_params[key] = v
            new_space = {k: v for k, v in space.items() if k != key}
            yield from parameter_generator(new_space, new_params)
    else:
        yield params


def remove_unnecessary_sequences(events):
    removed_indices = []
    for i in range(len(events) - 1, 0, -1):
        if events[i] == ['obscure'] and events[i - 1] == ['reveal']:
            removed_indices.extend([i - 1, i])
            del events[i - 1:i + 1]

    if events and events[-1] == ['reveal']:
        events.pop()

    # deal with swap indices being removed before swaps
    for event in events:
        if event[0] == 'swap':
            count = sum(1 for i in removed_indices if i < event[1])
            event[1] -= count
            if event[2] != 'empty':
                event[2] -= count

    return events


class ScenarioConfigs:
    tutorial = {
        "empty": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "a"
        },
        "empty_clutter": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "b"
        },
        "grid": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "c"
        },
        "grid_clutter": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "d"
        },
        "empty_hide": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "e"
        },
        "empty_hide_redherring": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "f"
        },
        "empty_hide_preference": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "g"
        },
        # 2 eVariants are all doorkey
        # 3 eVariants are also doorkey
        "nav_no_goal": {
            "num_puppets": 0,
            "eType": "n",
            "eVar": "a"
        },
        "nav_simple": {
            "num_puppets": 0,
            "eType": "n",
            "eVar": "b"
        },
        # ...
    }

    env_groups = {'12': ["stage_1", "stage_2"],
                  '1': ["stage_1"],
                  '2': ["stage_2"],
                  '3': ["informedControl",
                        "partiallyUninformed",
                        "removedInformed2",
                        "removedUninformed2",
                        "moved",
                        "replaced",
                        "misinformed",
                        "swapped",
                        ],
                  '3+12b': ["stage_1",
                            "s2b",
                            "informedControl",
                            "partiallyUninformed",
                            "removedInformed2",
                            "removedUninformed2",
                            "moved",
                            "replaced",
                            "misinformed",
                            "swapped",
                            ],
                  's2+s2b': ["stage_2", "s2b"],
                  '1+2b+all': ["all", "s2b", "stage_1"],
                  'all': ["all"],
                  's2b': ["s2b"],
                  'rand': ["random"],
                  'everything': ["stage_1",
                                 "s2b",
                                 "informedControl",
                                 "partiallyUninformed",
                                 "removedInformed2",
                                 "removedUninformed2",
                                 "moved",
                                 "replaced",
                                 "misinformed",
                                 "swapped",
                                 "all"
                                 ],
                  }

    so2 = {}

    count = 0

    parameter_space = {
        "visible_baits": range(3),  # x
        "swaps": range(3),  # y
        "visible_swaps": lambda params: range(params['swaps'] + 1),  # z
        "first_bait_size": [0, 1],  # i
        "uninformed_bait": lambda params: [0, 1] if params['visible_baits'] == 1 else [-1],  # k
        "uninformed_swap": lambda params: [0, 1] if params['swaps'] == 2 and params['visible_swaps'] == 1 else [-1],  # l
        "first_swap_is_both": lambda params: [True, False] if params['swaps'] > 0 else [False],  # ab
        "delay_2nd_bait": lambda params: [True, False] if params['swaps'] > 0 and params['first_swap_is_both'] is False else [False],  # d
        "second_swap_to_first_loc": lambda params: [True, False] if params['swaps'] == 2 and params['delay_2nd_bait'] is False else [False],  # c
        "first_swap": lambda params: [0, 1] if params['swaps'] > 0 and params['delay_2nd_bait'] is False and params['first_swap_is_both'] is False else [0],  # j
    }

    for params in parameter_generator(parameter_space):

        visible_baits = params['visible_baits']
        swaps = params['swaps']
        visible_swaps = params['visible_swaps']
        first_bait_size = params['first_bait_size']
        uninformed_bait = params['uninformed_bait']
        first_swap = params['first_swap']
        uninformed_swap = params['uninformed_swap']
        first_swap_is_both = params['first_swap_is_both']
        second_swap_to_first_loc = params['second_swap_to_first_loc']
        delay_2nd_bait = params['delay_2nd_bait']

        name = str(visible_baits) + "." + str(swaps) + "." + str(visible_swaps)
        if first_swap_is_both:
            name += "a"
        if second_swap_to_first_loc:
            name += "c"
        if delay_2nd_bait:
            name += "d"
        count += 1
        events = []

        print(name)

        bait_index = []
        for bait_num in range(1 if delay_2nd_bait else 2):
            bait_size = first_bait_size if bait_num == 0 else 1 - first_bait_size
            if bait_num == uninformed_bait or visible_baits == 0:
                events.extend(
                    [['obscure'], ['bait', bait_size, 'empty'], ['reveal']])
                bait_index.append(len(events) - 2)
            else:
                events.append(['bait', bait_size, 'empty'])
                bait_index.append(len(events) - 1)

            # Add swaps
        first_swap_index = None
        for swap_num in range(1 if delay_2nd_bait else swaps):
            if swap_num == 0 and first_swap_is_both:
                if swap_num == uninformed_swap or visible_swaps == 0:
                    events.extend([['obscure'], ['swap', bait_index[0], bait_index[1]], ['reveal']])
                    first_swap_index = len(events) - 2
                else:
                    events.append(['swap', bait_index[0], bait_index[1]])
                    first_swap_index = len(events) - 1

            else:
                swap_index = bait_index[first_swap] if swap_num == 0 else bait_index[1 - first_swap]
                if swap_num == 1 and second_swap_to_first_loc:
                    if swap_num == uninformed_swap or visible_swaps == 0:
                        events.extend([['obscure'], ['swap', swap_index, first_swap_index], ['reveal']])
                    else:
                        events.append(['swap', swap_index, first_swap_index])
                else:
                    if swap_num == uninformed_swap or visible_swaps == 0:
                        events.extend([['obscure'], ['swap', swap_index, 'empty'], ['reveal']])
                        if swap_num == 0:
                            first_swap_index = len(events) - 2
                    else:
                        events.append(['swap', swap_index, 'empty'])
                        if swap_num == 0:
                            first_swap_index = len(events) - 1

        if delay_2nd_bait:
            bait_size = 1 - first_bait_size
            bait_num = 1
            if bait_num == uninformed_bait or visible_baits == 0:
                events.extend([['obscure'], ['bait', bait_size, first_swap_index], ['reveal']])
                bait_index.append(len(events) - 2)
            else:
                events.append(['bait', bait_size, first_swap_index])
                bait_index.append(len(events) - 1)

        for swap_num in range(swap_num if delay_2nd_bait else 2, swaps):
            if swap_num == 0 and first_swap_is_both:
                if swap_num == uninformed_swap or visible_swaps == 0:
                    events.extend([['obscure'], ['swap', bait_index[0], bait_index[1]], ['reveal']])
                    first_swap_index = len(events) - 2
                else:
                    events.append(['swap', bait_index[0], bait_index[1]])
                    first_swap_index = len(events) - 1

            else:
                print(bait_index, first_swap, swap_num)
                swap_index = bait_index[first_swap] if swap_num == 0 else bait_index[1 - first_swap]
                if swap_num == 1 and second_swap_to_first_loc:
                    if swap_num == uninformed_swap or visible_swaps == 0:
                        events.extend([['obscure'], ['swap', swap_index, first_swap_index], ['reveal']])
                    else:
                        events.append(['swap', swap_index, first_swap_index])
                else:
                    if swap_num == uninformed_swap or visible_swaps == 0:
                        events.extend([['obscure'], ['swap', swap_index, 'empty'], ['reveal']])
                        if swap_num == 0:
                            first_swap_index = len(events) - 2
                    else:
                        events.append(['swap', swap_index, 'empty'])
                        if swap_num == 0:
                            first_swap_index = len(events) - 1

        events = remove_unnecessary_sequences(events)
        print(events)
    print('env count', count)

    standoff = {
        "defaults": {
            "hidden": True,
            "share_rewards": False,
            "boxes": 5,
            "sub_valence": 1,
            "dom_valence": 1,
            "subject_is_dominant": False,  # subordinate has delayed release. for subordinate first, use negative
            "num_puppets": 1,
            "num_agents": 1,
            "events": [[['bait', 'empty'], ['bait', 'empty']]]  # list, event, args
        },
        "random": {
            "num_puppets": [1],
            "events": [[['random']] * n for n in range(2, 6)]
        },
        "stage_1": {
            "num_puppets": [0],
            "boxes": [5],
        },
        "stage_2": {
            "num_puppets": [1],
            "boxes": [5],
        },
        "s2b": {
            "num_puppets": [1, 0],
        },
        "informedControl": {
            "events": [[['bait', 'empty'], ['bait', 'empty']]]  # optimal: Small
        },
        "partiallyUninformed": {  # optimal: if big bait obscured, Big, else Small
            "events": [[['bait', 'empty'], ['obscure'], ['bait', 'empty']],
                       [['obscure'], ['bait', 'empty'], ['reveal'], ['bait', 'empty']]]
        },
        "removedInformed1": {  # optimal: Neither (minor reward preference over copying dominant's decision)
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['remove', x]] for x in [0, 1]]
        },
        "removedUninformed1": {  # optimal: if Big is removed, Small, else Neither
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', x]] for x in [0, 1]]
        },
        "removedInformed2": {  # optimal: small
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['swap', x, 'empty']] for x in [0, 1]]
        },
        "removedUninformed2": {  # optimal: if Big is swapped, big, else small
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', x, 'empty']] for x in [0, 1]]
        },
        "moved": {  # optimal: Small
            "events": [[['obscure'], ['bait', 'empty'], ['bait', 'empty'], ['reveal'], ['swap', 1, 'empty'],
                        ['swap', 2, 'empty']]]
        },
        "replaced": {  # optimal: if first bait is big, Big, else Small
            "events": [[['bait', 'empty'], ['obscure'], ['swap', 0, 'empty'], ['bait', 0]]]
        },
        "misinformed": {
            # optimal: if big is swapped with empty, Big. If big swapped with small, also big. If small swapped with empty, small. If small swapped with big, big.
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', x, 'else'], ['swap', x, 'empty']]
                       for x in [0, 1]]
        },
        "swapped": {  # optimal: Big
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 1]]]
        },
        "all": {
            "events": [[['bait', 'empty'], ['bait', 'empty']],
                       [['bait', 'empty'], ['obscure'], ['bait', 'empty']],
                       [['obscure'], ['bait', 'empty'], ['reveal'], ['bait', 'empty']],
                       # [['bait', 'empty'], ['bait', 'empty'], ['remove', 0]],
                       # [['bait', 'empty'], ['bait', 'empty'], ['remove', 1]],
                       [['bait', 'empty'], ['bait', 'empty'], ['swap', 0, 'empty']],
                       [['bait', 'empty'], ['bait', 'empty'], ['swap', 1, 'empty']],

                       # [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', 0]],
                       # [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', 1]],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 'empty']],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 1, 'empty']],
                       [['obscure'], ['bait', 'empty'], ['bait', 'empty'], ['reveal'], ['swap', 1, 'empty'],
                        ['swap', 2, 'empty']],
                       [['bait', 'empty'], ['obscure'], ['swap', 0, 'empty'], ['bait', 0]],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 'else'], ['swap', 0, 'empty']],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 1, 'else'], ['swap', 1, 'empty']],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 1]]]
        }
    }
    standoff_optimal_policies = {
        "stage_1": 'big',
        "swapped": 'big',
        "partiallyUninformed": 'b-s',
        "replaced": 'b-s',
        "misinformed": 'b-s',
        "stage_2": 'small',
        "informedControl": 'small',
        "moved": 'small',
        "removedUninformed": 's-n',
        "removedInformed": 'none',
    }
