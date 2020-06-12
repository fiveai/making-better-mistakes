import os
import json
import re
import argparse
import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from glob import glob

NORMAL_MARKER = "o"
DEVISE_MARKER = "^"
XENT_MARKER = "*"
YOLO_MARKER = "d"
BD_MARKER = "s"
RANDOM_MARKER = "D"

FIG_SIZE = (6, 3.7)
FXE_COLOR = "g"
ALPHA = 1
# tab:orange
SFT_COLOR = (31 / 255, 119 / 255, 180 / 255, ALPHA)
# tab:blue
HXE_COLOR = (255 / 255, 127 / 255, 14 / 255, ALPHA)
NRM_COLOR = (0 / 255, 0 / 255, 0 / 255, ALPHA)

BASE_PATH = "../experiments"
OUTPUT_DIR = "./"

NSAMPLES = 5

experiment_to_best_epoch = dict()

# ------------------------------------------------------------------------- Tiered

HXENT_TIERED = [
    ("experiments/hxe_tieredimagenet_alpha0.1", "0.1"),
    ("experiments/hxe_tieredimagenet_alpha0.2", "0.2"),
    ("experiments/hxe_tieredimagenet_alpha0.3", "0.3"),
    ("experiments/hxe_tieredimagenet_alpha0.4", "0.4"),
    ("experiments/hxe_tieredimagenet_alpha0.5", "0.5"),
    ("experiments/hxe_tieredimagenet_alpha0.6", "0.6"),
]

SOFT_TIERED = [
    ("experiments/softlabels_tieredimagenet_beta04", "4"),
    ("experiments/softlabels_tieredimagenet_beta05", "5"),
    ("experiments/softlabels_tieredimagenet_beta10", "10"),
    ("experiments/softlabels_tieredimagenet_beta15", "15"),
    ("experiments/softlabels_tieredimagenet_beta20", "20"),
    ("experiments/softlabels_tieredimagenet_beta30", "30"),
]

XENT_TIERED = [
    ("experiments/crossentropy_tieredimagenet", ""),
]

DEVISE_TIERED = [("experiments/devise_tieredimagenet", "")]

YOLO_TIERED = [
    ("experiments/yolov2_tieredimagenet", ""),
]

BD_TIERED = [("experiments/barzdenzler_tieredimagenet", "")]

# ------------------------------------------------------------------------- iNaturalist19

HXENT_INAT19 = [
    ("experiments/hxe_inaturalist19_alpha0.1", "0.1"),
    ("experiments/hxe_inaturalist19_alpha0.2", "0.2"),
    ("experiments/hxe_inaturalist19_alpha0.3", "0.3"),
    ("experiments/hxe_inaturalist19_alpha0.4", "0.4"),
    ("experiments/hxe_inaturalist19_alpha0.5", "0.5"),
    ("experiments/hxe_inaturalist19_alpha0.6", "0.6"),
    ("experiments/hxe_inaturalist19_alpha0.7", "0.7"),
    ("experiments/hxe_inaturalist19_alpha0.9", "0.9"),
]

SOFT_INAT19 = [
    ("experiments/softlabels_inaturalist19_beta04", "4"),
    ("experiments/softlabels_inaturalist19_beta05", "5"),
    ("experiments/softlabels_inaturalist19_beta10", "10"),
    ("experiments/softlabels_inaturalist19_beta15", "15"),
    ("experiments/softlabels_inaturalist19_beta20", "20"),
    ("experiments/softlabels_inaturalist19_beta30", "30"),
]

XENT_INAT19 = [
    ("experiments/crossentropy_inaturalist19", ""),
]

YOLO_INAT19 = [("experiments/yolov2_inaturalist19", "")]

BD_INAT19 = [("experiments/barzdenzler_inaturalist19", "")]


def load_from_file(exp_path, section="val"):
    """Load all results for a given experiment as nested dict.

    Args:
        exp_path: The path of the experiment.
        section: Section of the results to get (train/val).

    Returns:
        Dictionary with all the results and the number of epochs.
    """

    json_path = os.path.join(exp_path, "json/")
    results = {}
    epochs = []

    # get and store name
    results["name"] = os.path.basename(os.path.normpath(exp_path))

    # load all json files
    paths = glob(os.path.join(json_path, section, "*"))
    for path in paths:
        m = re.match(r".*epoch\.(\d+)\.json", path)
        if m is not None:
            with open(path) as f:
                epoch = int(m.group(1))
                epochs.append(epoch)
                results[epoch] = json.load(f)
        else:
            raise RuntimeError("Can not match path ", path)

    assert len(epochs), "Could not load data for " + exp_path

    return results, list(sorted(epochs))


def get_optimal_epoch(results, epochs, min_epoch, max_epoch=None, plot_fit=False):
    """
    Get epoch that minimizes the loss.

    Args:
        results: The result dict from get_results().
        epochs: The list of epochs to use.

    Return:
        The optimal epoch number.
    """
    loss_key = [k for k in results[epochs[0]].keys() if k[:5] == "loss/"]
    assert len(loss_key) == 1, "Found multiple losses !"
    loss_key = loss_key[0]
    max_epoch = max_epoch or max(epochs)

    # use a polynomial fit
    fit_epochs = [e for e in epochs if min_epoch <= e <= max_epoch]
    fit_losses = [results[e][loss_key] for e in fit_epochs]
    poly = np.poly1d(np.polyfit(fit_epochs, fit_losses, 4))

    # get the minimum of the curve
    crit = poly.deriv().r
    r_crit = crit[crit.imag == 0].real
    test = poly.deriv(2)(r_crit)

    x_min = r_crit[test > 0]
    x_min = x_min[x_min <= max_epoch]
    if len(x_min) == 0:
        x_min = np.array([max_epoch])  # no solution means min is at the end
    if len(x_min) > 1:
        x_min = np.array([x_min[np.argmax(poly(x_min))]])  # two solutions: take the min
    if poly(max_epoch) < poly(x_min):
        x_min = x_min[np.argmax(poly(x_min))]  # consider boundary as well
    y_min = poly(x_min)

    if plot_fit:
        # plot fit line for visual check
        plt.figure()
        plt.plot(fit_epochs, fit_losses)
        plt.plot(fit_epochs, poly(fit_epochs))
        plt.plot(x_min, y_min, "ro")
        plt.savefig(os.path.join(OUTPUT_DIR, results["name"] + "-loss-fit.pdf"), bbox_inches="tight")
        plt.close()

    # find index of epoch closest to the min
    index = np.argmin([abs(e - x_min) for e in epochs])

    return index


@functools.lru_cache(maxsize=None)
def get_results(path, nsamples, min_epoch, max_epoch=None, plot_fit=False):
    """
    Load results for a given path and find the epoch range.

    Args:
        path: The path (relative to base directory) where the results are.
        nsamples: Number of samples to use for computing mean+-std.
        min_epoch, max_epoch: The min and max epoch to perform the fit.
    """
    print("\nLoading results from {}".format(path))
    results, epochs = load_from_file(os.path.join(BASE_PATH, path))

    if max_epoch is not None:
        epochs = [e for e in epochs if e <= max_epoch]
    nepochs = len(epochs) - 1
    print("Found {} epochs.".format(nepochs))

    # get index of the best epoch
    best = get_optimal_epoch(results, epochs, min_epoch, max_epoch, plot_fit=plot_fit)

    print("Best epoch is {}".format(best))

    # compute start and end indices
    end = best + nsamples // 2
    if end > nepochs:
        end = nepochs
    start = end - nsamples + 1

    print("Selecting epoch range [{}, {}].".format(epochs[start], epochs[end]))

    return results, epochs, start, end


def load_and_plot_experiments(
        experiments,
        process_results,
        yaxis,
        xaxis,
        nsamples,
        color,
        marker,
        section="val",
        min_epoch=0,
        max_epoch=None,
        hollow_marker=False,
        show_labels=True,
        plot_fit=False,
):
    """
    Load results and plot list of experiments.

    Args:
        experiments: Dictionary {name: path}, see example above.
        yaxis: The metric to use as yaxis.
        xaxis: The metric to use as xaxis.
        nsamples: Number of samples to use for computing mean+-std.
        section: The section of the results to use.
        min_epoch, max_epoch: The min and max epochs to perform the fit.
    """

    x_all = []
    y_all = []

    # remove alpha for marker labels
    if len(color) == 4:
        text_color = color[:-1]
    else:
        text_color = color

    # In these cases the polynomial fails to capture the loss behaviour over the full range of epochs, so we harcode the range were the optimum is.
    for path, label in experiments:
        if path == "experiments/softlabels_inaturalist19_beta15":
            min_epoch = 0

        results, epochs, start, end = get_results(path, nsamples, min_epoch, max_epoch, plot_fit=plot_fit)

        # write list of epochs around minimum on dictionary
        if path not in experiment_to_best_epoch:
            experiment_to_best_epoch[path] = [epochs[e] for e in range(start, end + 1)]
        else:
            assert experiment_to_best_epoch[path] == [epochs[e] for e in range(start, end + 1)], "Found two different best epoch for run '{}'".format(path)

        x_values = [process_results["x"](results[epochs[i]][xaxis]) for i in range(start, end + 1)]
        y_values = [process_results["y"](results[epochs[i]][yaxis]) for i in range(start, end + 1)]

        x_m = np.median(x_values)
        # x_e = np.std(x_values, ddof=1)
        y_m = np.median(y_values)
        # y_e = np.std(y_values, ddof=1)

        x_all.append(x_m)
        y_all.append(y_m)

        if not hollow_marker:
            plt.plot(x_m, y_m, color=color, marker=marker, zorder=100)
        else:
            plt.plot(x_m, y_m, color=color, marker=marker, zorder=100, markerfacecolor="w")

        if show_labels:
            plt.text(x_m * (1 - 0.01), y_m * (1 - 0.01), label, color=text_color, fontsize=8)

    plt.plot(x_all, y_all, "--", color="k", alpha=0.4, zorder=0, linewidth=1)


def save_figure(name, yaxis, xaxis):
    fig_name = "{}_{}_{}.pdf".format(name, xaxis.replace("/", "."), yaxis.replace("/", "."))
    fig_name = os.path.join(OUTPUT_DIR, fig_name)
    print("Saving figure {}".format(fig_name))
    plt.savefig(fig_name, bbox_inches="tight")


def make_figures(process_results, axes, legend_location, plot_fit=False):
    # ############################ #
    # main / tiered                #
    # ############################ #

    legend_elements = [
        Line2D([], [], marker=NORMAL_MARKER, color=SFT_COLOR, label="Soft labels", linestyle="None"),
        Line2D([], [], marker=NORMAL_MARKER, color=HXE_COLOR, label="HXE", linestyle="None"),
        Line2D([], [], marker=XENT_MARKER, color=NRM_COLOR, label="Cross-entropy", linestyle="None"),
        Line2D([], [], marker=YOLO_MARKER, color=NRM_COLOR, label="YOLO-v2", linestyle="None"),
        Line2D([], [], marker=BD_MARKER, color=NRM_COLOR, label=r"Barz \& Denzler", linestyle="None"),
        Line2D([], [], marker=DEVISE_MARKER, color=NRM_COLOR, label="DeViSE", linestyle="None"),
    ]

    fig = plt.figure(figsize=FIG_SIZE)

    plt.legend(handles=legend_elements, loc=legend_location, facecolor="white", framealpha=1, edgecolor="k", fancybox=False)

    load_and_plot_experiments(
        HXENT_TIERED,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=HXE_COLOR,
        marker=NORMAL_MARKER,
        min_epoch=25,
        max_epoch=115,
        plot_fit=plot_fit,
    )
    load_and_plot_experiments(
        SOFT_TIERED,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=SFT_COLOR,
        marker=NORMAL_MARKER,
        min_epoch=25,
        max_epoch=115,
        plot_fit=plot_fit,
    )
    load_and_plot_experiments(
        XENT_TIERED,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=NRM_COLOR,
        marker=XENT_MARKER,
        min_epoch=25,
        max_epoch=115,
        plot_fit=plot_fit,
    )
    load_and_plot_experiments(
        YOLO_TIERED,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=NRM_COLOR,
        marker=YOLO_MARKER,
        min_epoch=25,
        max_epoch=115,
        plot_fit=plot_fit,
    )
    load_and_plot_experiments(
        DEVISE_TIERED,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=NRM_COLOR,
        marker=DEVISE_MARKER,
        min_epoch=25,
        max_epoch=115,
        plot_fit=plot_fit,
    )
    load_and_plot_experiments(
        BD_TIERED,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=NRM_COLOR,
        marker=BD_MARKER,
        min_epoch=25,
        max_epoch=115,
        plot_fit=plot_fit,
    )

    plt.xlabel(axes["xlabel"])
    plt.ylabel(axes["ylabel"])
    save_figure("tiered-imagenet-h", axes["y"], axes["x"])
    plt.close(fig)

    # ############################ #
    # main / iNat19                #
    # ############################ #

    legend_elements = legend_elements[:-1]

    fig = plt.figure(figsize=FIG_SIZE)

    plt.legend(handles=legend_elements, loc=legend_location, facecolor="white", framealpha=1, edgecolor="k", fancybox=False)

    load_and_plot_experiments(
        HXENT_INAT19,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=HXE_COLOR,
        marker=NORMAL_MARKER,
        min_epoch=80,
        max_epoch=270,
        plot_fit=plot_fit,
    )
    load_and_plot_experiments(
        SOFT_INAT19,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=SFT_COLOR,
        marker=NORMAL_MARKER,
        min_epoch=80,
        max_epoch=270,
        plot_fit=plot_fit,
    )
    load_and_plot_experiments(
        XENT_INAT19,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=NRM_COLOR,
        marker=XENT_MARKER,
        min_epoch=80,
        max_epoch=270,
        plot_fit=plot_fit,
    )
    load_and_plot_experiments(
        YOLO_INAT19,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=NRM_COLOR,
        marker=YOLO_MARKER,
        min_epoch=80,
        max_epoch=270,
        plot_fit=plot_fit,
    )
    load_and_plot_experiments(
        BD_INAT19,
        process_results,
        yaxis=axes["y"],
        xaxis=axes["x"],
        nsamples=NSAMPLES,
        color=NRM_COLOR,
        marker=BD_MARKER,
        max_epoch=270,
        plot_fit=plot_fit,
    )

    plt.xlabel(axes["xlabel"])
    plt.ylabel(axes["ylabel"])
    save_figure("inaturalist19", axes["y"], axes["x"])
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results.")
    parser.add_argument("--exp-path", default=BASE_PATH, help="Folder with the experiment data")
    parser.add_argument("--output-dir", default="../paper_main_plots", help="Folder to dump the plots.")
    parser.add_argument("--plot-fit", action="store_true", help="Also plot fit curves.")
    parser.add_argument("--big", action="store_true", help="Produce bigger plots.")
    parser.add_argument("--latex", action="store_false", help="Use latex for text.")
    opts = parser.parse_args()

    if not opts.big:
        FIG_SIZE = (5, 3.1)

    if opts.latex:
        params = {
            "text.usetex": True,
            "font.size": 10,
            "font.family": "lmodern",
            "text.latex.unicode": True,
        }
        plt.rcParams.update(params)

    BASE_PATH = opts.exp_path
    OUTPUT_DIR = opts.output_dir
    plot_fit = opts.plot_fit

    # Pass lambda functions to use for processing the results before plotting
    process_results = dict()
    process_results["x"] = lambda x: 100 * (1 - x)
    process_results["y"] = lambda y: y

    axes = dict()
    axes["x"] = "accuracy_top/01"
    axes["xlabel"] = "Top-1 error"

    axes["y"], axes["ylabel"] = "ilsvrc_dist_mistakes/avg01", "Hier. dist. mistake"
    make_figures(process_results, axes, "best", plot_fit=plot_fit)
    axes["y"], axes["ylabel"] = "ilsvrc_dist_avg/01", "Avg. hier. dist. @1"
    make_figures(process_results, axes, "best", plot_fit=plot_fit)
    axes["y"], axes["ylabel"] = "ilsvrc_dist_avg/05", "Avg. hier. dist. @5"
    make_figures(process_results, axes, "upper right", plot_fit=plot_fit)
    axes["y"], axes["ylabel"] = "ilsvrc_dist_avg/20", "Avg. hier. dist. @20"
    make_figures(process_results, axes, "upper right", plot_fit=plot_fit)

    with open("experiment_to_best_epoch.json", "w") as fp:
        json.dump(experiment_to_best_epoch, fp)
