import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import dflat.plot_utilities as df_plt

dirname = str(Path(__file__).parent) + "/"


def compute_statistics(sorted_data):
    num_pts = len(sorted_data)
    fquart = sorted_data[int(num_pts * 0.25)]
    median = sorted_data[int(num_pts * 0.50)]
    thirdquart = sorted_data[int(num_pts * 0.75)]

    return [fquart, median, thirdquart]
    # return [sorted_data[0], fquart, median, thirdquart, sorted_data]


def load_model_get_stat(path_to_model, dat_string, stat_fun):

    dat_path = path_to_model + "/training_testDataError.pickle"
    file = open(dat_path, "rb")
    data = pickle.load(file)
    file.close()

    # account for polarization vs nonpolarization arrays
    thisdat = data[dat_string]
    if len(thisdat.shape) == 1:
        thisdat = np.expand_dims(thisdat, 0)

    abs_error = np.mean(np.abs(thisdat), axis=0)
    absError_sort = np.sort(abs_error)

    dat_stat = stat_fun(absError_sort)
    log_flops = np.log10(data["est_FLOPs"])

    return dat_stat, log_flops


def sweep_files_stats(fold_path, file_name_list, stat_fun, dat_string="complex_error"):

    stat_array = []
    log_flop_array = []
    for file_name in file_name_list:
        stat, log_flops = load_model_get_stat(fold_path + file_name, dat_string, stat_fun=stat_fun)
        stat_array.append(stat)
        log_flop_array.append(log_flops)

    return np.stack(stat_array), np.stack(log_flop_array)


def generate_nanofin_benchmark_figure():
    savepath = dirname

    ### Define Data Paths
    trained_mlp_path = dirname + "/model_dat/trained_MLP_models/"
    mlp_model_names = [
        # "MLP_Nanofins_Dense128_U350_H600",
        "MLP_Nanofins_Dense256_U350_H600",
        "MLP_Nanofins_Dense512_U350_H600",
        "MLP_Nanofins_Dense1024_U350_H600",
    ]

    fitted_multipoly_path = dirname + "model_dat/trained_MultiPoly_models/"
    multipoly_model_names = ["multipoly_nanofins_6", "multipoly_nanofins_12", "multipoly_nanofins_18", "multipoly_nanofins_24"]

    trained_erbf_path = dirname + "model_dat/trained_erbf_models/"
    erbf_model_names = ["ERBF_Nanofins_B512_U350_H600", "ERBF_Nanofins_B1024_U350_H600", "ERBF_Nanofins_B4096_U350_H600", "ERBF_Nanofins_B5000_U350_H600"]

    ### Grab Model Statistics
    mlp_stats, mlp_log_flops = sweep_files_stats(trained_mlp_path, mlp_model_names, compute_statistics, dat_string="complex_error")
    poly_stats, poly_log_flops = sweep_files_stats(fitted_multipoly_path, multipoly_model_names, compute_statistics, dat_string="complex_error")
    erbf_stats, erbf_log_flops = sweep_files_stats(trained_erbf_path, erbf_model_names, compute_statistics, dat_string="complex_error")

    ### Make Figure
    dataXPlot = [mlp_log_flops, poly_log_flops, erbf_log_flops]
    dataYPlot = [mlp_stats, poly_stats, erbf_stats]
    color = ["green", "red", "blue"]

    fig = plt.figure()
    ax = df_plt.addAxis(fig, 1, 1)
    for i in range(len(dataXPlot)):
        xdat = dataXPlot[i]
        ydat = dataYPlot[i]
        useColor = color[i]
        for j in range(ydat.shape[1]):
            ax[0].plot(xdat, ydat[:, j], "x-", c=useColor, alpha=1.0)
        ax[0].fill_between(xdat, ydat[:, 1], ydat[:, 0], color=useColor, alpha=0.3)
        ax[0].fill_between(xdat, ydat[:, 2], ydat[:, 1], color=useColor, alpha=0.3)

    plt.grid(color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    legend_elements = [
        Line2D([0], [0], color="g", lw=3, label="MLP"),
        Line2D([0], [0], color="r", lw=3, label="Multivariate Polynomial"),
        Line2D([0], [0], color="b", lw=3, label="Axis-aligned ERBF"),
    ]
    ax[0].legend(handles=legend_elements)
    plt.savefig(savepath + "model_comp.png")
    plt.savefig(savepath + "model_comp.pdf")

    return


if __name__ == "__main__":
    generate_nanofin_benchmark_figure()
    plt.show()
