import matplotlib.pyplot as plt
import numpy as np


def plot_distribution():
    s =0.1
    model = 'GraphSage'
    dataset = 'DD'
    plt.figure(figsize=(6,5))
    scaler_0_path = f'out_150/{dataset}_{model}/result/scaler/{s}.npy'
    data = np.load(scaler_0_path)
    member = data[:len(data) // 2]
    nonmember = data[len(data) // 2:]

    print("Member data range:", np.min(member), np.max(member))
    print("Non-member data range:", np.min(nonmember), np.max(nonmember))

    alpha = 0.7
    bins = 30
    color1 = 'salmon'
    color2 = 'royalblue'
    edgecolor = 'black'

    plt.hist(member, bins=bins, alpha=alpha, label='member', color=color1, edgecolor=edgecolor)
    plt.hist(nonmember, bins=bins, alpha=alpha, label='non-member', color=color2, edgecolor=edgecolor)

    plt.xlabel('Robustness Score')
    # plt.title(f'{dataset}  s={s}')
    plt.legend()
    plt.xticks([0.0,0.1, 0.2, 0.3,0.4,0.5, 0.6, 0.7,0.8, 0.9,1.0])

    plt.xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(f'out_once150/{dataset}_{model}/result/scaler/{dataset}_{model}_s={s}.png')
    plt.gca().set_yticks([])
    plt.show()

plot_distribution()


