import numpy as np

class NArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.q_true = np.random.rand(n_arms)  # Zufällige Gewinnwahrscheinlichkeiten für jede Maschine

    def pull(self, arm):
        return 1 if np.random.rand() < self.q_true[arm] else 0  # Gewinn (1) oder Verlust (0)

# Beispiel mit 5-Armed Bandit
n_arms = 5
bandit = NArmedBandit(n_arms)

# Ziehe zufällig einen Hebel
chosen_arm = np.random.randint(n_arms)
reward = bandit.pull(chosen_arm)

print(f"Gewählte Maschine: {chosen_arm}")
print(f"Belohnung: {reward}")
