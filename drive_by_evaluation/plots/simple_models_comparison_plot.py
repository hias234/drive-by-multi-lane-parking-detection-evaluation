import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = 5

accuracy =               (0.507796, 0.932811, 0.949871, 0.959449, 0.93)
precision_parked_cars =  (   0.457,    0.804,    0.876,    0.908, 0.92)
recall_parked_cars =     (   0.888,    0.834,    0.860,    0.880, 0.92)
learning_time =          (    0.14,    34.02,     0.65,     6.38, 0.92)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.6
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, accuracy, bar_width,
                #alpha=opacity,
                color='#002984',
                label='Accuracy')

rects2 = ax.bar(index + bar_width, precision_parked_cars, bar_width,
                #alpha=opacity,
                color='#00675b',
                label='Precision Parked Cars')

rects3 = ax.bar(index + bar_width * 2, recall_parked_cars, bar_width,
                #alpha=opacity,
                color='#c8b900',
                label='Recall Parked Cars')

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                "{0:.2f}".format(height),
                ha='center', va='bottom',
                fontsize=6)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# ax.set_xlabel('Group')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Naive Bayes', 'Neural Network', 'Decision Tree', 'Random Forest', 'SVC'))
ax.legend()

fig.tight_layout()
plt.show()