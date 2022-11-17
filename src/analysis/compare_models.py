import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import auc
from scipy.stats import sem

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
    output_png_zoom = 'dataset_dependent/panda/experiments/final_experiments/final_results_zoomlarge.png'
    model_dirs = ['bnn_ra', 'bnn_en',  'bnn_bald', 'bnn_ms', 'bnn_ep',  'focal', ]
    model_names = ['RA', 'EN', 'BALD', 'MS', 'EP', 'FocAL', ]
    colors = ['tab:blue',
              'tab:orange',
              'tab:green',
              'tab:red',
              'tab:purple',
              'k']
    # model_dirs = ['bnn_bald', 'focal']
    # model_names = ['BALD',  'FocAL']
    # colors = ['k',
    #           'tab:pink']
    #
    # model_dirs = ['bnn_ra',  'bnn_bald', 'bnn_ms', 'bnn_ep',  'focal', ]
    # model_names = ['RA', 'BALD', 'MS', 'EP', 'FocAL']
    # colors = ['tab:blue',
    #           'tab:green',
    #           'tab:red',
    #           'tab:purple',
    #           'k']

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
        std = sem(np.array(cols_of_interest), axis=1)
        ax.errorbar(steps, mean, yerr=std, color=colors[i],  marker=".",zorder = zorders[i],
                    label=model_names[i])
    auc_m = auc(steps.astype(np.int), mean)
    # print(model_names[i] + ' AUC: ' + str(auc_m))
    print(model_names[i] + ' Final mean K: ' + str(mean[-1]) + ' SE: ' + str(std[-1]))
if table == 'abl':
    ax.set_xticks(ax.get_xticks()[::2])
else:
    ax.set_xticks(steps)
    plt.ylim([0.45, 0.775])

plt.ylabel('Cohens quadratic kappa')
plt.xlabel('Labeled image patches')
plt.legend(loc='upper left')
plt.savefig(output_png, bbox_inches='tight')

if table == 'sota':
    # min_index = 2
    # df = df.iloc[min_index:,:]
    steps = np.array(df.index)
    plt.figure()
    fig, ax = plt.subplots(figsize=(3.7, 2.5))
    for i in range(int(len(df.columns) / num_runs)):
        column = i * num_runs
        cols_of_interest = df.iloc[:, column:column + num_runs]
        mean = np.array(np.mean(cols_of_interest, axis=1))
        std = sem(np.array(cols_of_interest), axis=1)
        ax.errorbar(steps, mean, yerr=std, color=colors[i], marker=".", zorder=zorders[i],
                    label=model_names[i])
    ax.set_xticks(steps)
    plt.ylim([0.65, 0.77])
    plt.xlim([1000, 4500])
    ax.set_xticks([])
    plt.savefig(output_png_zoom, bbox_inches='tight')


