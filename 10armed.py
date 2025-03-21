import numpy as np

class TenArmedTestbed:
    def __init__(self):
        self.n_arms = 10
        # Ziehe die wahren Erwartungswerte aus einer Normalverteilung N(0,1)
        self.q_true = np.random.randn(self.n_arms)

    def pull(self, arm):
        # Generiere eine Belohnung als Sample aus einer Normalverteilung
        # mit Mittelwert q_true[arm] und Standardabweichung 1.
        return np.random.randn() + self.q_true[arm]

# Beispielnutzung:
testbed = TenArmedTestbed()

# Ziehe zufällig einen Hebel
chosen_arm = np.random.randint(testbed.n_arms)
reward = testbed.pull(chosen_arm)

print(f"Gewählte Maschine: {chosen_arm}")
print(f"Belohnung: {reward}")
