import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 5000

temperature = np.random.normal(50, 10, n_samples)
gas_level = np.random.normal(40, 15, n_samples)
pressure = np.random.normal(100, 20, n_samples)

maintenance_active = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
shift_change = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])

risk_label = []

for i in range(n_samples):
    risk = 0

    if gas_level[i] > 60:
        risk += 1

    if pressure[i] > 130:
        risk += 1

    if temperature[i] > 70:
        risk += 1

    if maintenance_active[i] == 1:
        risk += 1

    risk_label.append(1 if risk >= 2 else 0)

df = pd.DataFrame({
    "temperature": temperature,
    "gas_level": gas_level,
    "pressure": pressure,
    "maintenance_active": maintenance_active,
    "shift_change": shift_change,
    "risk_label": risk_label
})

df.to_csv("data/industrial_safety_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())