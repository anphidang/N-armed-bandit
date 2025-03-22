import numpy as np

class NArmedBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_true = np.random.rand(n_arms)  # Wahre Gewinnwahrscheinlichkeiten
        self.q_estimated = np.zeros(n_arms)   # Geschätzte Q-Werte (Action-Values)
        self.n_selected = np.zeros(n_arms)    # Anzahl Ziehungen pro Arm

    def select_arm(self):
        if np.random.rand() < self.epsilon:  # Mit Wahrscheinlichkeit ε: Zufällig wählen (Exploration)
            return np.random.randint(self.n_arms)
        else:  # Mit Wahrscheinlichkeit 1 - ε: Besten Arm wählen (Exploitation)
            return np.argmax(self.q_estimated)

    def update(self, arm, reward):
        self.n_selected[arm] += 1
        self.q_estimated[arm] += (reward - self.q_estimated[arm]) / self.n_selected[arm]  # Inkrementelles Mittel

    def pull(self, arm):
        return 1 if np.random.rand() < self.q_true[arm] else 0  # Gewinn oder Verlust

# Beispiel mit 5-Armed Bandit und 1000 Versuchen
n_arms = 5
epsilon = 0.1
bandit = NArmedBandit(n_arms, epsilon)

n_steps = 1000
rewards = []

for _ in range(n_steps):
    arm = bandit.select_arm()
    reward = bandit.pull(arm)
    bandit.update(arm, reward)
    rewards.append(reward)

print("Geschätzte Q-Werte:", bandit.q_estimated)
print("Wahre Q-Werte:", bandit.q_true)
print("Anzahl der Ziehungen pro Arm:", bandit.n_selected)
print("Durchschnittliche Belohnung:", np.mean(rewards))
