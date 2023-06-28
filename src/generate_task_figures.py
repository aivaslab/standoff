from .pz_envs import scenario_configs
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def line_state(line):
    line_string = ''
    for symbol in line:

        symbol_type = symbol['char']
        line_string += symbol_type + ' '
    return line_string

def prev_state(line):
    line_string = ''
    for i, symbol in enumerate(line):
        prev = symbol['prev']
        if prev == None:
            line_string += '  '
        else:
            if prev == i:
                line_string += '| '
            elif prev < i:
                line_string += '\ '
            elif prev > i:
                line_string += '/ '
    return line_string

def update_line_state(line, update, all_positions):
    returned_position = 0
    update_type = update[0]
    update_arg = update[1] if len(update) > 1 else None
    update_arg2 = update[2] if len(update) > 2 else None
    for i, symbol in enumerate(line):
        if symbol['char'] == 'o':
            symbol['char'] = 'x'
        symbol['prev'] = i
    if update_type == 'b':
        if update_arg == 'e':
            # find first item in line where char is x
            for i, symbol in enumerate(line):
                if symbol['char'] == '.':
                    symbol['char'] = 'o'
                    returned_position = i
                    break
            line[returned_position]['prev'] = None
        else:
            index = all_positions[update_arg]
            line[index]['prev'] = None
            line[index]['char'] = 'o'
    elif update_type == 'remove':
        index = all_positions[update_arg]
        line[index]['char'] = ' '
        line[index]['prev'] = None
    elif update_type == 'sw':
        index1 = all_positions[update_arg]
        if update_arg2 == 'else':
            index2 = (index1 + 2) % 5
        elif update_arg2 == 'e':
            for i, symbol in enumerate(line):
                if symbol['char'] == '.':
                    index2 = i
                    break
        else:
            index2 = all_positions[update_arg2]
        line[index1]['char'], line[index2]['char'] = line[index2]['char'], line[index1]['char']
        # update prev to be position of previous data before swap
        line[index1]['prev'] = index2
        line[index2]['prev'] = index1
    elif update_type == 'rel':
        # change all empty strings to .s, all prevs to index
        for i, symbol in enumerate(line):
            if symbol['char'] == ' ':
                symbol['char'] = '.'
            symbol['prev'] = i
        
    return line, returned_position

'''
#ascii version
for name in stageNames[4]:
    print(name)
    cur_line = [{'char': '.', 'prev': None} for _ in range(5)]
    all_positions = []
    updates = standoff[name]['events'][0] if 'events' in standoff[name] else standoff['defaults']['events'][0]
    timestep = 0
    print('\t' + str(timestep) + ': '+ line_state(cur_line))
    for update in updates:
        timestep += 1
        cur_line, pos = update_line_state(cur_line, update, all_positions)
        print('\t' + ' ' + '  '+ prev_state(cur_line) + str(update))
        print('\t' + str(timestep) + ': '+ line_state(cur_line)) 
        all_positions.append(pos)
    cur_line, pos = update_line_state(cur_line, (('release', 0)), all_positions)
    print('\t'  + ' ' + '  '+ prev_state(cur_line) + str(('release')))
    print('\t' + '   '+ line_state(cur_line)) 
'''


stageNames, standoff = reset_names()
# create a new figure
major_updates = []
updateses = []
names = []
for name in stageNames[4]:
    major_updates.append(standoff[name]['events'] if 'events' in standoff[name] else standoff['defaults']['events'])
for mu, name in zip(major_updates, stageNames[4]):
    for x in range(len(mu)):
        updateses.append(mu[x])
        names.append(name + ' ' + str(x))
fig, axs = plt.subplots(len(updateses), figsize=(6, 50))

for i, (updates, name) in enumerate(zip(updateses, names)):
    cur_line = [{'char': '.', 'prev': None} for _ in range(5)]
    all_positions = []
    timestep = 0
    updates.append(('release', 0))

    ax = axs[i]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')

    ax.set_title(name)
    ax_offset = 0.25
    max_size = len(updates)
    obscured = False

    for update in updates:
        timestep += 1
        cur_line, pos = update_line_state(cur_line, update, all_positions)
        if update[0] == 'ob':
            obscured = True
        if update[0] == 're' or update[0] == 'rel':
            obscured = False

        all_positions.append(pos)
        prev_data = []
        for i, x in enumerate(cur_line):
            if x['char'] != '.' and x['char'] != 'x':
                ax.text(i-0.1, ax_offset + max_size - timestep - 2, x['char'], size='xx-large')
            prev_data.append((x['prev'], i))

        for prev, curr in prev_data:
            if prev is not None:
                if prev != curr:
                    ax.arrow(prev, ax_offset + max_size-(timestep+1), (curr - prev)*1, -1, length_includes_head=True, head_width=0.2, head_length=0.2, fc='k', ec='k', lw=2)
                else:
                    ax.arrow(prev, ax_offset + max_size-(timestep+1), (curr - prev), -1, length_includes_head=True, head_width=0.0, head_length=0.0, fc='k', ec='k', lw=2)

        if obscured:
            ax.add_patch(Rectangle((-0.5, ax_offset + max_size - timestep - 2 - 0.25), 5.5, 1, alpha=0.2, facecolor='grey'))