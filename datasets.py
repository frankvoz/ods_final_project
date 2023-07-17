import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


ALPHA_CH = 0.3

colors = ['deepskyblue', 'orange']
cmap = ListedColormap(colors)


def _reformat(data_orig, col_predictors):
    data = data_orig.copy()

    for i in col_predictors:
        data[i] = data[i].astype(str)
        # data[i] = data[i].apply(lambda x: x[2:])
        data[i] = data[i].apply(lambda x: x[x.index(':')+1:] if ':' in x else x)
        data[i] = data[i].astype(float)

    data = data.fillna(0)
    data.head()
    return data

def _convert_y(y):
    value = y[0]
    return np.where(y == value, 1, -1)

def _svd(X):
    # Xc = X - np.mean(X, axis=0)
    U, _, _ = np.linalg.svd(X)

    u1 = U[:,1]
    u2 = U[:,2]
    u3 = U[:,3]
    return u1, u2, u3

def plot_2D(X, y, cmap=cmap, alpha=ALPHA_CH):
    u1, u2, _ = _svd(X)

    plt.scatter(u1, u2, c=y, cmap=cmap, alpha=alpha)

def plot_3D(X, y, cmap=cmap, alpha=ALPHA_CH):
    u1, u2, u3 = _svd(X)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(u1, u2, u3, c=y, cmap=cmap, alpha=alpha)
    plt.show()

'''
def load_dataset(path):
    data_orig = pd.read_csv(path, sep=' ', header=None)

    column_to_move = 0
    columns = data_orig.columns.tolist()

    if data_orig.iloc[:, -1].isnull().any():
        columns.pop() # remove nan column
        
    columns.remove(column_to_move)
    col_predictors = columns
    data_orig = data_orig.reindex(columns=columns+[column_to_move])

    data = _reformat(data_orig, col_predictors)

    X = data[col_predictors].to_numpy()
    y = data[0].to_numpy()

    y = _convert_y(y)

    return X, y
'''

def load_dataset(path):
    ll = []

    with open(path, 'r') as file:
        line = file.readline().strip()
        d = int(line)

        columns = [f'Feature_{i}' for i in range(1, d)]
        columns.append('Target')

        for line in file:
            words = line.strip().split()

            target = words[0]

            f_list = []
            v_list = []
            row = []

            # Process the feature-value pairs
            for word in words[1:]:
                feature, value = word.split(':')
                f_list.append(int(feature))
                v_list.append(float(value))
            
            for i in range(1, d):
                if i in f_list:
                    idx = f_list.index(i)
                    row.append(v_list[idx])
                else:
                    row.append(np.nan)
            row.append(int(target))
            ll.append(row)

    data = pd.DataFrame(ll, columns=columns)
    data = data.dropna(thresh=0.5*len(data), axis=1)
    data = data.fillna(data.median())

    X = data[data.columns[:-1]].to_numpy()
    y = data[data.columns[-1]].to_numpy()

    y = _convert_y(y)

    return X, y