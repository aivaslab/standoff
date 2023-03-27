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

    env_groups = {0: ["stage_1", "stage_2"],
                  1: ["stage_1"],
                  2: ["stage_2"],
                  3: ["informed control",
                      "partially uninformed",
                      "removed informed",
                      "removed uninformed",
                      "moved",
                      "replaced",
                      "misinformed",
                      "swapped",
                      ],
                  4: ["stage_1",
                      "stage_2",
                      "informed control",
                      "partially uninformed",
                      "removed informed",
                      "removed uninformed",
                      "moved",
                      "replaced",
                      "misinformed",
                      "swapped",
                      ],
                  }

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
            "events": [[['bait', 'empty'], ['bait', 'empty']]]  # optimal: Small
        },
        "partially uninformed": {  # optimal: if big bait obscured, Big, else Small
            "events": [[['bait', 'empty'], ['obscure'], ['bait', 'empty']],
                       [['obscure'], ['bait', 'empty'], ['reveal'], ['bait', 'empty']]]
        },
        "removed informed": {  # optimal: Neither (minor reward preference over copying dominant's decision)
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['remove', x]] for x in [0, 1]]
        },
        "removed uninformed": {  # optimal: if Big is removed, Small, else Neither
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', x]] for x in [0, 1]]
        },
        "moved": {  # optimal: Small
            "events": [[['obscure'], ['bait', 'empty'], ['bait', 'empty'], ['reveal'], ['swap', 1, 'empty'],
                        ['swap', 2, 'empty']]]
        },
        "replaced": {  # optimal: if first bait is big, Big, else Small
            "events": [[['bait', 'empty'], ['obscure'], ['swap', 0, 'empty'], ['bait', 0]]]
        },
        "misinformed": {  # optimal: if big is swapped, Big, else Small
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', x, 'else'], ['swap', 3, x]] for x in [0, 1]]
        },
        "swapped": {  # optimal: Big
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
    standoff_optimal_policies = {
        "stage_1": 'big',
        "swapped": 'big',
        "partially uninformed": 'b-s',
        "replaced": 'b-s',
        "misinformed": 'b-s',
        "stage_2": 'small',
        "informed control": 'small',
        "moved": 'small',
        "removed uninformed": 's-n',
        "removed informed": 'none',
    }
