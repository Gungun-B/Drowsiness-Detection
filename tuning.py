import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ear_log.csv")
t0 = df["timestamp"].iloc[0]
df["t"] = df["timestamp"] - t0

plt.figure(figsize=(10, 5))
plt.plot(df["t"], df["ear"], label="EAR")
plt.plot(df["t"], df["mar"], label="MAR")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("EAR and MAR over time")
plt.legend()
plt.grid(True)
plt.show()

drowsy = df["status"] == "DROWSY"
plt.scatter(df["t"][drowsy], df["ear"][drowsy], color="red", s=10, label="Drowsy EAR")
plt.legend()
plt.show()
