def remove_unnecessary_sequences(events):
    removed_indices = []
    for i in range(len(events) - 1, 0, -1):
        if events[i] == ['obscure'] and events[i - 1] == ['reveal']:
            removed_indices.extend([i-1, i])
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
    for visible_baits in [0, 1, 2]:  # x
        for swaps in [0, 1, 2]:  # y
            for visible_swaps in range(swaps + 1):  # z
                for first_bait_size in [0, 1]:  # i
                    for uninformed_bait in [0, 1] if visible_baits == 1 else [-1]:  # k
                        for first_swap in [0, 1] if swaps > 0 else [-1]:  # j
                            for uninformed_swap in [0, 1] if swaps == 2 and visible_swaps == 1 else [-1]:  # l
                                for first_swap_is_both in [True, False] if swaps > 0 else [False]:  # ab
                                    for second_swap_to_first_loc in [True, False] if swaps == 2 else [False]:  # c
                                        for delay_2nd_bait in [True, False] if swaps > 0 else [False]:  # d
                                            name = str(visible_baits) + "." + str(swaps) + "." + str(visible_swaps)
                                            if first_swap_is_both:
                                                name += "a"
                                            if second_swap_to_first_loc:
                                                name += "c"
                                            if delay_2nd_bait:
                                                name += "d"
                                            count += 1
                                            events = []

                                            bait_index = []
                                            for bait_num in range(2):
                                                bait_size = first_bait_size if bait_num == 0 else 1 - first_bait_size
                                                if bait_num == uninformed_bait or visible_baits == 0:
                                                    events.extend([['obscure'], ['bait', bait_size, 'empty'], ['reveal']])
                                                    bait_index.append(len(events) - 2)
                                                else:
                                                    events.append(['bait', bait_size, 'empty'])
                                                    bait_index.append(len(events) - 1)

                                                # Add swaps
                                            for swap_num in range(swaps):
                                                swap_index = bait_index[first_swap] if swap_num == 0 else bait_index[1 - first_swap]
                                                if swap_num == uninformed_swap or visible_swaps == 0:
                                                    events.extend( [['obscure'], ['swap', swap_index, 'empty'], ['reveal']])
                                                else:
                                                    events.append(['swap', swap_index, 'empty'])

                                            events = remove_unnecessary_sequences(events)
                                            print(name, events)
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
