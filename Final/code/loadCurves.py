import numpy as np
import pandas as pd
from scipy import io


def create_nasa_curves():
    """Creates a matrix of all NASA battery curves, and its labels

    Creates each curve as a row in in a large X matrix. Each curve is voltage
    interpolated to be in 128 steps as a function of the power delivered from
    2% to 98%.

    """
    battery_files = ['B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat']

    X = None
    p = np.linspace(.02, .98, 128)
    labels = None
    capacity = None

    for bat_file in battery_files:
        bat_path = 'Final/data/battery/' + bat_file
        matlab_file = io.loadmat(bat_path, simplify_cells=True)
        bat_str = list(matlab_file.keys())[-1]
        label = int(bat_str[1:])
        
        discharge_cycles = [cycle for cycle in matlab_file[bat_str]['cycle']
                            if cycle['type'] == 'discharge']

        N = len(discharge_cycles)
        X_bat = np.zeros((N, len(p)))
        l_bat = np.zeros(N, dtype=np.int8)
        cap_bat = np.zeros(N)

        for i, cycle in enumerate(discharge_cycles):
            df = pd.DataFrame(cycle['data'])

            power = -(df['Voltage_measured'])*(df['Current_measured'])
            dt = np.roll(df['Time'], -1) - df['Time']
            dt[0] = 0
            energy = np.cumsum(power*dt)
            energy_frac = energy/np.max(energy)
            V = np.interp(p, energy_frac, df['Voltage_measured'])
            X_bat[i, :] = V
            l_bat[i] = label
            cap_bat[i] = np.max(energy)

        cap_bat /= np.max(cap_bat)

        X = X_bat if X is None else np.append(X, X_bat, axis=0)
        labels = l_bat if labels is None else np.append(labels, l_bat)
        capacity = cap_bat if capacity is None else np.append(capacity, cap_bat)

    np.save('Final/output/X.npy', X)
    np.save('Final/output/p.npy', p)
    np.save('Final/output/labels.npy', labels)
    np.save('Final/output/capacity.npy', capacity)


def load_nasa_curves(to_load):
    """Loads the curves cached to disk

    Returns:
        X (ndarray): Matrix of curves
        p (ndarray): The discharge percentage that X rows are a function of
        labels (ndarray): Whuch battery each curve is
        capacity (ndarray): The capacity of each battery during each curve
    """
    X = np.load('Final/output/X.npy')
    p = np.load('Final/output/p.npy')
    labels = np.load('Final/output/labels.npy')
    capacity = np.load('Final/output/capacity.npy')

    mask = np.isin(labels, to_load)
    return X[mask], p, labels[mask], capacity[mask]


if __name__ == '__main__':
    create_nasa_curves()
