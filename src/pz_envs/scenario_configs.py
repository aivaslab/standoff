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

    stageNames = {1: ["stage_1"],
                  2: ["stage_2"],
                  3: ["informed control",
                      "partially uninformed",
                      "removed informed",
                      "removed uninformed",
                      "moved",
                      "replaced",
                      "misinformed",
                      "swapped",
                      "all"
                      ]}

    standoff = {
        "defaults": {
            "hidden": True,
            "share_rewards": False,
            "boxes": 5,
            "sub_valence": 1,
            "dom_valence": 1,
            "subject_is_dominant": False,  # subordinate has delayed release. for subordinate first, use negative
            "lava_height": 3,  # should be odd, >=2
            "num_puppets": 1,
            "num_agents": 1,
            "events": [[['bait', 'empty'], ['bait', 'empty']]]  # list, event, args
        },

        "stage_1": {
            "num_puppets": [0],
            "boxes": [5],
        },
        "stage_2": {
            "num_puppets": [1],
            "boxes": [5],
        },
        "informed control": {
            "events": [[['bait', 'empty'], ['bait', 'empty']]]
        },
        "partially uninformed": {
            "events": [[['bait', 'empty'], ['obscure'], ['bait', 'empty']],
                       [['obscure'], ['bait', 'empty'], ['reveal'], ['bait', 'empty']]]
        },
        "removed informed": {
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['remove', x]] for x in [0, 1]]
        },
        "removed uninformed": {
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', x]] for x in [0, 1]]
        },
        "moved": {
            "events": [[['obscure'], ['bait', 'empty'], ['bait', 'empty'], ['reveal'], ['swap', 1, 'empty'],
                        ['swap', 2, 'empty']]]
        },
        "replaced": {
            "events": [[['bait', 'empty'], ['obscure'], ['swap', 0, 'empty'], ['bait', 0]]]
        },
        "misinformed": {
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', x, 'else']] for x in [0, 1]]
        },
        "swapped": {
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 1]]]
        },
        "all": {
            "events": [[['bait', 'empty'], ['bait', 'empty']],
                       [['bait', 'empty'], ['obscure'], ['bait', 'empty']],
                       [['obscure'], ['bait', 'empty'], ['reveal'], ['bait', 'empty']],
                       [['bait', 'empty'], ['bait', 'empty'], ['remove', 0]],
                       [['bait', 'empty'], ['bait', 'empty'], ['remove', 1]],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', 0]],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', 1]],
                       [['obscure'], ['bait', 'empty'], ['bait', 'empty'], ['reveal'], ['swap', 1, 'empty'],
                        ['swap', 2, 'empty']],
                       [['bait', 'empty'], ['obscure'], ['swap', 0, 'empty'], ['bait', 0]],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 'else']],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 1, 'else']],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 1]]]
        }
    }


class AllParams():
    params = {"standoffEnv": {
        "adversarial": [True, False],
        "hidden": [True, False],
        "rational": [True, False],
        "share_rewards": [True, False],
        "firstBig": [True, False],  # whether we place big first
        "boxes": [2, 3, 4, 5],
        "num_puppets": [0, 1, 2],
        "followDistance": [0, 1],  # 0 = d first, 1=sub first
        "lava_height": [2],
        "baits": [1, 2],
        "baitSize": [1, 2],
        "informed": ['informed', 'uninformed', 'fake', 'half1', 'half2'],
        "swapType": ['swap', 'replace', 'remove', 'move', 'mis'],
        "visibility": ['full', 'curtains'],  # keys, invisibility potion
        "cause": ['blocks', 'direction', 'accident', 'inability'],
        "lava": ['lava', 'block'],
        "num_agents": [1, 2],
    }
    }
