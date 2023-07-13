import os

from src.supervised_learning import gen_data
from supervised_learning_main import run_supervised_session


def experiments(repetitions, epochs, todo):
    """What is the overall performance of naive, off-the-shelf models on this task? Which parameters of competitive
    feeding settings are the most sensitive to overall model performance? To what extent are different models
    sensitive to different parameters? """

    regimes = [
        ('situational', ['a1']),
        ('informed', ['i0', 'i1']),
        ('contrastive', ['i0', 'u0', 'i1', 'u1']),
        ('complete', ['a0', 'i1']),
        ('comprehensive', ['a0', 'i1', 'u1'])
    ]
    default_regime = regimes[1]
    pref_types = [
        ('same', ''),
        ('different', 'd'),
        ('varying', 'v'),
    ]
    role_types = [
        ('subordinate', ''),
        ('dominant', 'D'),
        ('varying', 'V'),
    ]

    # generate supervised data
    if 0 in todo:
        for pref_type, pref_suffix in pref_types:
            for role_type, role_suffix in role_types:
                gen_data(['correctSelection'], path='supervised', pref_type=pref_suffix, role_type=role_suffix)

    if 'h' in todo:
        print('Running hyperparameter search on all regimes, pref_types, role_types')


    # Experiment 1
    if 1 in todo:
        print('Running experiment 1: varied models training directly on the test set')


        # todo: add hparam search for many models, comparison between them?
        run_supervised_session(save_path=os.path.join('supervised', 'exp_1'),
                               repetitions=repetitions,
                               epochs=epochs,
                               train_sets=['a1'])

    # Experiment 2
    if 2 in todo:
        print('Running experiment 2: varied train regimes')
        for regime, train_sets in regimes:
            # todo: add hparam search for each regime. for test set accuracy?
            print('regime:', regime)
            run_supervised_session(save_path=os.path.join('supervised', 'exp_2', regime),
                                   repetitions=repetitions,
                                   epochs=epochs,
                                   train_sets=train_sets)

    # Experiment 3
    if 3 in todo:
        print('Running experiment 3: varied preferences')
        for pref_type, pref_suffix in pref_types:
            for regime, train_sets in [default_regime]:
                new_train_sets = [x + pref_suffix for x in train_sets]
                run_supervised_session(save_path=os.path.join('supervised', 'exp_3', pref_type, regime),
                                       repetitions=repetitions,
                                       epochs=epochs,
                                       train_sets=new_train_sets)

    # Experiment 4
    if 4 in todo:
        print('Running experiment 4: varied role')
        for role_type, role_suffix in role_types:
            for pref_type, pref_suffix in pref_types:
                for regime, train_sets in [default_regime]:
                    new_train_sets = [x + pref_suffix + role_suffix for x in train_sets]
                    run_supervised_session(save_path=os.path.join('supervised', 'exp_4', role_type, pref_type, regime),
                                           repetitions=repetitions,
                                           epochs=epochs,
                                           train_sets=new_train_sets)

    # Experiment 5
    if 5 in todo:
        print('Running experiment 5: varied collaboration')

    # Experiment 100
    if 100 in todo:
        print('Running experiment -1: testing effect of dense vs sparse inputs')
        # todo: add hparam search for many models, comparison between them?
        run_supervised_session(save_path=os.path.join('supervised', 'exp_100'),
                               repetitions=repetitions,
                               epochs=epochs,
                               train_sets=['a1'])


if __name__ == '__main__':
    experiments(1, 30, [2])