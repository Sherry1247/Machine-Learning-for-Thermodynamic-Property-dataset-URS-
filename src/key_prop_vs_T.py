import re
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

file_path = "data/C-095.txt"

with open(file_path, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

start_idx = next(i for i, l in enumerate(lines) if re.match(r"^\d", l))
data_text = "\n".join(lines[start_idx:])

# columns
cols = ["T_K", "Cp", "S", "GminusH_over_T", "HminusH_Tr",
        "deltaf_H", "deltaf_G", "logKf"]

df = pd.read_csv(StringIO(data_text), sep=r"\s+", header=None,
                 names=cols, na_values=["INFINITE", "INFINITY"])

print("First 5 rows of data:")

plt.figure(figsize=(10,6))
print(df.head())
plt.plot(df["T_K"], df["Cp"], label="Cp (J·K⁻¹·mol⁻¹)")
plt.plot(df["T_K"], df["S"], label="Entropy S (J·K⁻¹·mol⁻¹)")
plt.plot(df["T_K"], df["HminusH_Tr"], label="H - H° (kJ/mol)")
plt.plot(df["T_K"], df["deltaf_H"], label="ΔfH (kJ/mol)")
plt.plot(df["T_K"], df["deltaf_G"], label="ΔfG (kJ/mol)")

plt.xlabel("Temperature (K)")
plt.ylabel("Thermodynamic values")
plt.title("CO₂ — Thermodynamic Properties vs Temperature")
plt.legend()
plt.grid(True)
plt.show()
