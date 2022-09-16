import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import auc


input_csv = 'dataset_dependent/panda/experiments/ablation_studies/results.csv'
output_png = 'dataset_dependent/panda/experiments/ablation_studies/ablation_studies.png'
model_names = ['$\lambda_{al}$:0.5 $\lambda_{od}$:0.5',
               '$\lambda_{al}$:0.5 $\lambda_{od}$:1.0',
               '$\lambda_{al}$:1.0 $\lambda_{od}$:0.5',
               '$\lambda_{al}$:1.0 $\lambda_{od}$:1.0',
               '$\lambda_{al}$:1.0 $\lambda_{od}$:2.0',
               '$\lambda_{al}$:2.0 $\lambda_{od}$:1.0',
               '$\lambda_{al}$:2.0 $\lambda_{od}$:0.5',
               '$\lambda_{al}$:0.5 $\lambda_{od}$:2.0',
               '$\lambda_{al}$:2.0 $\lambda_{od}$:2.0']

colors = ['tab:blue',
          'k',
          'tab:orange',
          'tab:green',
          'tab:red',
          'tab:purple',
          'tab:olive',
          'tab:cyan',
          'tab:pink',
          ]
zorders = [0,10,0,0,0,0,0,0,0]



df = pd.read_csv(input_csv).T.dropna(axis=1)
# set first row as header
df.columns = df.iloc[0]
df = df[1:]
df.head()

steps = np.array(df.index)
fig, ax = plt.subplots()

for i in range(int(len(df.columns)/3)):
    column = i*3
    mean = np.array(np.mean(df.iloc[:,column:column+3], axis=1))
    ax.plot(steps, mean, color=colors[i],  marker=".",zorder = zorders[i], label=model_names[i])
    auc_m = auc(steps.astype(np.int),mean)
    print(model_names[i] + ' AUC: ' + str(auc_m))
ax.set_xticks(ax.get_xticks()[::2])
plt.ylabel('Cohens quadratic kappa')
plt.xlabel('Labeled image patches')
plt.legend()
plt.savefig(output_png)


