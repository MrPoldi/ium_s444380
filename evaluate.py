from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

test_data = pd.read_csv("out.csv")

y_true = test_data["OFFENSE_CODE_GROUP"]
y_pred = test_data["PREDICTED"]


accuracy = accuracy_score(y_true, y_pred)

with open("eval_results.csv", "a", encoding="utf-8") as f:
    f.write(f"{accuracy}\n")

eval_results = pd.read_csv("eval_results.csv", header=None).values

plt.plot(np.arange(len(eval_results)), eval_results)
plt.xlabel("Build")
plt.ylabel("Accuracy")
plt.savefig("plot.png")
