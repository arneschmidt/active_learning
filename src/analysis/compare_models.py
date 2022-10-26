import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import auc

table = 'sota' # 'abl' or 'sota'


if table == 'abl':
    input_csv = 'dataset_dependent/panda/experiments/ablation_studies/results.csv'
    output_png = 'dataset_dependent/panda/experiments/ablation_studies/ablation_studies.png'
    model_names = ['$\lambda_{al}$:0.5 $\lambda_{ood}$:0.5',
                   '$\lambda_{al}$:0.5 $\lambda_{ood}$:1.0',
                   '$\lambda_{al}$:1.0 $\lambda_{ood}$:0.5',
                   '$\lambda_{al}$:1.0 $\lambda_{ood}$:1.0',
                   '$\lambda_{al}$:1.0 $\lambda_{ood}$:2.0',
                   '$\lambda_{al}$:2.0 $\lambda_{ood}$:1.0',
                   '$\lambda_{al}$:2.0 $\lambda_{ood}$:0.5',
                   '$\lambda_{al}$:0.5 $\lambda_{ood}$:2.0',
                   '$\lambda_{al}$:2.0 $\lambda_{ood}$:2.0']

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
    num_runs = 3
else:
    input_dir = '/home/arne/projects/active_learning/experiment_output/mlflow_artifacts/'
    output_png = 'dataset_dependent/panda/experiments/final_experiments/final_results.png'
    model_dirs = ['bnn_bald', 'bnn_en', 'bnn_ep', 'bnn_ms', 'focal']
    model_names = ['BALD', 'EN', 'EP', 'MS', 'FocAL']
    colors = ['tab:blue',
              'tab:orange',
              'tab:green',
              'tab:red',
              'k']
    zorders = [0, 0, 0, 0, 0, 10]
    num_runs = 3
    max_index = 11
    df = pd.DataFrame
    for m in range(len(model_dirs)):
        for r in range(num_runs):
            model_run = model_dirs[m] + '_' + str(r)
            path = input_dir + model_run + '/results.csv'
            run_df = pd.read_csv(path)
            if m == 0  and r == 0:
                run_df.rename( columns={'Unnamed: 0': 'labeled_images'}, inplace=True )
                df = pd.DataFrame({'labeled_images': run_df['labeled_images']})
            test_kappa = run_df['test_cohens_quadratic_kappa']
            # test_kappa = run_df['test_f1_mean']
            df[model_run] = test_kappa
    df = df.iloc[0:max_index,:]
    df = df.set_index(df.columns[0])
    # df = df.drop(['labeled_images'], axis=1)

steps = np.array(df.index)
fig, ax = plt.subplots()

for i in range(int(len(df.columns)/num_runs)):
    column = i*num_runs
    cols_of_interest = df.iloc[:,column:column+num_runs]
    mean = np.array(np.mean(cols_of_interest, axis=1))
    if table == 'abl':
        ax.plot(steps, mean, color=colors[i],  marker=".",zorder = zorders[i], label=model_names[i])
    else:
        std = np.std(np.array(cols_of_interest), axis=1)
        ax.errorbar(steps, mean, yerr=std, marker=".",
                    label=model_names[i])
    auc_m = auc(steps.astype(np.int), mean)
    print(model_names[i] + ' AUC: ' + str(auc_m))
ax.set_xticks(ax.get_xticks()[::2])
plt.ylabel('Cohens quadratic kappa')
plt.xlabel('Labeled image patches')
plt.legend()
plt.savefig(output_png)


