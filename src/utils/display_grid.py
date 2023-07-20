import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()

# Define the range of the x and y axes
x_range = [0, 1, 2]  # visible_baits
y_range = [0, 1, 2]  # swaps

cell_sizes = {0: 1, 1: 0.5, 2: 0.33}
cell_offsets = [0.005, 0.01, 0.015]

# Create the grid
for x in x_range:
    for y in y_range:
        # Outer rectangle
        rect = Rectangle((x, y), 1, 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Subdivision for "visible_swaps"
        for z in range(y + 1):
            # Sub rectangle
            sub_rect = Rectangle((x + cell_offsets[0], y + z*cell_sizes[y] + cell_offsets[0]), 1 - 2*cell_offsets[0], cell_sizes[y]-2*cell_offsets[0], fill=False, edgecolor='blue', linewidth=1)
            ax.add_patch(sub_rect)
            ax.text(3 + cell_offsets[0], y + z * cell_sizes[y] + cell_offsets[0], f'visible_swaps={z}', color='blue', fontsize=8)

            sub_rect = Rectangle((x + cell_offsets[0], y + z * cell_sizes[y] + cell_offsets[0]),
                                 1 - 2 * cell_offsets[0], cell_sizes[y] - 2 * cell_offsets[0], fill=True,
                                 edgecolor='blue', linewidth=1)

            if (y == 0):
                if (x == 2 and z == 0 ):
                    ax.text(x + 0.33 + 0.14, y + z * cell_sizes[y] + 0.2, 'ic', color='black', fontsize=8,
                            ha='left')
                    ax.add_patch(sub_rect)
                if (x == 1 and z == 0 ):
                    ax.text(x + 0.33 +0.14, y + z * cell_sizes[y] + 0.2, 'pu', color='black', fontsize=8,
                            ha='left')
                    ax.add_patch(sub_rect)

            # Subdivision for "first_swap_is_both" and delay_2nd
            if y > 0:
                for w in [0, 1, 2]:
                    sub_rect2 = Rectangle((x + 0.33*w + cell_offsets[1], y + z*cell_sizes[y] + cell_offsets[1]), 0.33 - 2*cell_offsets[1], cell_sizes[y] - 2*cell_offsets[1], fill=False, edgecolor='red', linewidth=1)

                    if w == 1:
                        ax.text(x + 0.33*w + 0.14, 3.05, 'fsb', color='red', fontsize=8, ha='left')
                    elif w == 2:
                        ax.text(x + 0.33*w + 0.14, 3.05, 'dsb', color='red', fontsize=8, ha='left')

                    ax.add_patch(sub_rect2)

                    sub_rect2 = Rectangle((x + 0.33 * w + cell_offsets[1], y + z * cell_sizes[y] + cell_offsets[1]),
                                          0.33 - 2 * cell_offsets[1], cell_sizes[y] - 2 * cell_offsets[1],
                                          fill=True, edgecolor='red', linewidth=1)
                    if (y==0):
                        if (x==2 and z==0 and w==0):
                            ax.text(x + 0.33*w + 0.14, y + z*cell_sizes[y]+0.2, 'ic', color='black', fontsize=8, ha='left')
                            ax.add_patch(sub_rect2)
                        if (x==1 and z==0 and w==0):
                            ax.text(x + 0.33*w + 0.14, y + z*cell_sizes[y]+0.2, 'pu', color='black', fontsize=8, ha='left')
                            ax.add_patch(sub_rect2)

                    if (y==1):

                        if (x==1 and z==0 and w==2): #replaced,
                            ax.text(x + 0.33*w + 0.14, y + z*cell_sizes[y]+0.2, 're', color='black', fontsize=8, ha='left')
                            ax.add_patch(sub_rect2)
                        if (x==2 and z==0 and w==0): #ru,
                            ax.text(x + 0.33*w + 0.14, y + z*cell_sizes[y]+0.2, 'ru', color='black', fontsize=8, ha='left')
                            ax.add_patch(sub_rect2)
                        if (x==2 and z==0 and w==1): #sw,
                            ax.text(x + 0.33*w + 0.14, y + z*cell_sizes[y]+0.2, 'sw', color='black', fontsize=8, ha='left')
                            ax.add_patch(sub_rect2)
                        if (x==2 and z==1 and w==0): #ri
                            ax.text(x + 0.33*w + 0.14, y + z*cell_sizes[y]+0.2, 'ri', color='black', fontsize=8, ha='left')
                            ax.add_patch(sub_rect2)

                    if (y == 2 and w != 2):  # condition for "second_swap_to_first_loc"
                        for sw in [0, 1]:
                            sub_rect3 = Rectangle((x + 0.33*w + cell_offsets[2], y + z*cell_sizes[y] + sw*cell_sizes[y]/2 + cell_offsets[2]), 0.33 - 2* cell_offsets[2], cell_sizes[y]/2 - 2*cell_offsets[2], fill=False, edgecolor='green', linewidth=1)
                            ax.add_patch(sub_rect3)
                            ax.text(2 + 0.5, y + 1.05, 'second_swap_to_first_loc', color='green', fontsize=8, ha='center')

                            if (z == 2 and x == 0 and sw == 0 and w == 0): #moved
                                sub_rect3 = Rectangle((x + 0.33 * w + cell_offsets[2],
                                                       y + z * cell_sizes[y] + sw * cell_sizes[y] / 2 + cell_offsets[
                                                           2]), 0.33 - 2 * cell_offsets[2],
                                                      cell_sizes[y] / 2 - 2 * cell_offsets[2], fill=True,
                                                      edgecolor='green', linewidth=1)
                                ax.add_patch(sub_rect3)
                                ax.text(x + 0.33 * w + cell_offsets[2] + 0.2,
                                                       y + z * cell_sizes[y] + sw * cell_sizes[y] / 2 + cell_offsets[
                                                           2], 'mo', color='black', fontsize=8, ha='center')

                            if (z == 0 and x == 2 and sw == 1): #misinformed
                                sub_rect3 = Rectangle((x + 0.33 * w + cell_offsets[2],
                                                       y + z * cell_sizes[y] + sw * cell_sizes[y] / 2 + cell_offsets[
                                                           2]), 0.33 - 2 * cell_offsets[2],
                                                      cell_sizes[y] / 2 - 2 * cell_offsets[2], fill=True,
                                                      edgecolor='green', linewidth=1)
                                ax.add_patch(sub_rect3)
                                ax.text(x + 0.33 * w + cell_offsets[2] + 0.2,
                                                       y + z * cell_sizes[y] + sw * cell_sizes[y] / 2 + cell_offsets[
                                                           2], 'mi', color='black', fontsize=8, ha='center')



ax.set_xlim(0, len(x_range))
ax.set_ylim(0, len(y_range))
ax.set_xticks([0.5, 1.5, 2.5])
ax.set_yticks([0.5, 1.5, 2.5])
ax.set_xticklabels(['visible_baits=0', 'visible_baits=1', 'visible_baits=2'])
ax.set_yticklabels(['swaps=0', 'swaps=1', 'swaps=2'])

plt.show()