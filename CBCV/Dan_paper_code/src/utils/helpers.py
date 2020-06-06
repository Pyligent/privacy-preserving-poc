import pickle
import os
import random

import numpy as np

def get_data(data_dir, window_size):

    data = pickle.load(open(os.path.join(data_dir, 'data_april_24.pkl'), 'rb'))
    quarterly_revenue = data['quarterly_data']
    monthly_customers = np.asarray(data['panel_data'])
    # agg data size = (# of staggered rows, window_size + 1)
    agg_data = []
    for i in range(0, len(quarterly_revenue) - window_size):
        agg_data.append(quarterly_revenue[i: i + window_size + 1])
    agg_data = np.array(agg_data)
    # panel data
    panel_data = []
    for i in range(0, agg_data.shape[0]):
        panel_data.append(monthly_customers[list(range(i,i+(window_size * 3)))])
    panel_data = np.array(panel_data)

    # Rows are customers, columns are months, then quarters
    combined_data = np.hstack([panel_data, agg_data])
    indicies = get_index_split(
        {'train' : 0.8, 'val' : 0.1, 'test' : 0.1},
        set(range(combined_data.shape[0])),
    )
    data = {}
    for phase in ['train', 'val', 'test']:
        data['X_' + phase] = combined_data[indicies[phase],:-1]
        data['y_' + phase] = combined_data[indicies[phase],-1]
    return data

def get_index_split(ratios, index_set):
    counts = {}
    counts['val'] = round(len(index_set) * ratios['val'])
    counts['test'] = round(len(index_set) * ratios['test'])
    counts['train'] = len(index_set) - counts['test'] - counts['val']
    indicies = {}
    for phase in ['train', 'test', 'val']:
        indicies[phase] = random.sample(index_set, counts[phase])
        index_set = index_set - set(indicies[phase])
    return indicies
