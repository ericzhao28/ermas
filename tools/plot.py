import matplotlib.pyplot as plt

for k in history:
    plt.plot(history[k])
    plt.title(k)
    plt.show()

import matplotlib.pyplot as plt


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


for k in history:
    # x = np.arange(len(data))
    # y = data
    if not len(history[k]):
        print("Skipping", k)
        continue
    if "reward" in k:
        data = moving_average(history[k], 500)
    else:
        data = moving_average(history[k], 10000)
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.plot(np.arange(len(data)), data)
    plt.title(k)
    plt.show()
