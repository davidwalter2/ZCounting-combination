import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

from utils import logging, plot_tools, styles

logger = logging.child_logger(__name__)


def plot_pulls(params, names, postfix=None, outDir="./", markersize=4, cms_decor=None):
    """
    nuisance parameters pulls and constraints
    """

    logger.info("Make pulls plot")

    xx = np.array([p["correlated_value"].n for p in params])
    xx_hi = np.array([p["correlated_value"].s for p in params])
    xx_lo = np.array([p["correlated_value"].s for p in params])
    yy = np.arange(len(names))

    # parameters with a name starting with "r_" are rate parameters
    is_rate = np.array([True if n.startswith("r_") else False for n in names])

    names = np.array([styles.translate.get(n, n) for n in names])

    fig, ax = plt.subplots()
    fig.subplots_adjust(hspace=0.0, left=0.4, right=0.97, top=0.97, bottom=0.125)

    ymin = yy[0] - 1
    ymax = yy[-1] + 1

    nround = lambda x: round(x, 3)

    for i, r in enumerate(is_rate):
        if r:
            ax.text(
                -0.5,
                yy[i] - 0.4,
                "$"
                + str(nround(xx[i]))
                + "^{+"
                + str(nround(xx_hi[i]))
                + "}_{"
                + str(nround(xx_lo[i]))
                + "}$",
                bbox=dict(facecolor="white", edgecolor="none"),
                zorder=2,  # Lower z-order than the spines
            )
    for spine in ax.spines.values():
        spine.set_zorder(3)

    # only plot nuisance parameters that are constraint (no rate parameters)
    xx = xx[~is_rate]
    xx_hi = xx_hi[~is_rate]
    xx_lo = xx_lo[~is_rate]
    yy = yy[~is_rate]

    ax.plot([1, 1], [ymin, ymax], linestyle="dashed", color="gray")
    ax.plot([-1, -1], [ymin, ymax], linestyle="dashed", color="gray")

    ax.plot([2, 2], [ymin, ymax], linestyle="dashed", color="gray")
    ax.plot([-2, -2], [ymin, ymax], linestyle="dashed", color="gray")

    ax.plot([0.0, 0.0], [ymin, ymax], linestyle="dashed", color="gray")

    ax.errorbar(
        xx,
        yy,
        xerr=(abs(xx_lo), abs(xx_hi)),
        fmt="ko",
        ecolor="black",
        elinewidth=1.0,
        capsize=1.0,
        barsabove=True,
        markersize=markersize,
    )
    if cms_decor is not None:
        hep.cms.label(label=cms_decor, loc=0, ax=ax, data=True)

    ax.set_xlabel("($\\hat{\\Theta} - \\Theta_0 ) / \\Delta \\Theta$")
    ax.set_ylabel("")

    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(ymin, ymax)

    name = "pulls"

    if postfix is not None:
        name += f"_{postfix}"

    plot_tools.save_plot(outDir, name)


def plot_pulls_lumi(
    df,
    df_blue=None,
    chi2_info=None,
    outDir="./",
    xRange=(-0.018, 0.018),
    markersize=4,
    cms_decor=None,
):
    """
    plot results on luminosity
    """

    logger.info("Make plot of lumi results")

    if df_blue is not None:
        custom_order = ["2017H", "2016", "2017", "2018", "Sum"]
        df["era"] = pd.Categorical(df["era"], categories=custom_order, ordered=True)
        df = df.sort_values(by="era")

    translate = {
        "2017H": "Low PU",
    }

    names = [translate.get(n, n) for n in df["era"].values]
    yy = np.arange(len(names))

    xx_prefit = df["prefit"].values
    xx_hi_prefit = df["prefit_uncertainty"].values / xx_prefit
    xx_lo_prefit = xx_hi_prefit

    xx_hi_prefit = df["prefit_uncertainty"].values / xx_prefit
    xx_lo_prefit = xx_hi_prefit

    xx = (df["postfit"].values - xx_prefit) / xx_prefit
    xx_hi = df["postfit_uncertainty"].values / xx_prefit
    xx_lo = xx_hi

    if df_blue is not None:
        xx_prefit_z = (df_blue["prefit_z"].values - xx_prefit[1:]) / xx_prefit[1:]
        xx_hi_prefit_z = df_blue["prefit_z_uncertainty"].values / xx_prefit[1:]
        xx_lo_prefit_z = xx_hi_prefit_z

        xx_blue = (df_blue["postfit"].values - xx_prefit[1:]) / xx_prefit[1:]
        xx_hi_blue = df_blue["postfit_uncertainty"].values / xx_prefit[1:]
        xx_lo_blue = xx_hi_blue

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(hspace=0.0, left=0.4, right=0.97, top=0.97, bottom=0.125)

    ymin = yy[0] - 1
    ymax = yy[-1] + 1

    # ax.plot([-0.02, -0.02], [ymin, ymax], linestyle="dashed", color="gray")
    # ax.plot([0.02, 0.02], [ymin, ymax], linestyle="dashed", color="gray")
    ax.plot([-0.01, -0.01], [ymin, ymax], linestyle="dashed", color="gray")
    ax.plot([0.01, 0.01], [ymin, ymax], linestyle="dashed", color="gray")
    ax.plot([0.0, 0.0], [ymin, ymax], linestyle="dashed", color="gray")

    # nround = lambda x: round(x, 3)
    # for i, r in enumerate(is_rate):
    #     if r:
    #         ax.text(-0.5, yy[i]-0.4, "$"+str(nround(xx[i]))+"^{+"+str(nround(xx_hi[i]))+"}_{"+str(nround(xx_lo[i]))+"}$")

    if df_blue is not None:
        # vertical line to separate 2017H
        ax.plot(xRange, [0.5, 0.5], linestyle="dashed", color="gray")

        yy_ref = yy + 0.3
        yy_z = yy[1:] + 0.1
        yy_blue = yy[1:] - 0.1
        yy_nll = yy - 0.3

        ax.errorbar(
            xx_prefit_z,
            yy_z,
            xerr=(abs(xx_lo_prefit_z), abs(xx_hi_prefit_z)),
            label="Z lumi",
            marker="o",
            color="black",
            fillstyle="none",
            linestyle="none",
            elinewidth=1.0,
            capsize=1.0,
            barsabove=True,
            markersize=markersize,
        )
        ax.errorbar(
            xx_blue,
            yy_blue,
            xerr=(abs(xx_lo_blue), abs(xx_hi_blue)),
            label="BLUE combination",
            marker="o",
            color="blue",
            elinewidth=1.0,
            linestyle="none",
            capsize=1.0,
            barsabove=True,
            markersize=markersize,
        )
    else:
        yy_ref = yy + 0.2
        yy_nll = yy - 0.2

    ax.errorbar(
        np.zeros_like(xx_prefit),
        yy_ref,
        xerr=(abs(xx_lo_prefit), abs(xx_hi_prefit)),
        label="Ref. luminosity",
        marker="o",
        linestyle="none",
        color="black",
        elinewidth=1.0,
        capsize=1.0,
        barsabove=True,
        markersize=markersize,
    )

    ax.errorbar(
        xx,
        yy_nll,
        xerr=(abs(xx_lo), abs(xx_hi)),
        label="Likelihood combination",
        marker="o",
        linestyle="none",
        color="red",
        elinewidth=1.0,
        capsize=1.0,
        barsabove=True,
        markersize=markersize,
    )

    # if df_blue is None:
    #     for y, x, x_lo, x_prefit, x_lo_prefit in zip(
    #         yy, xx, xx_lo, xx_prefit, xx_lo_prefit
    #     ):
    #         x = (x * x_prefit + x_prefit) / 1000.0
    #         x_lo = (x_lo * x_prefit) / 1000.0
    #         x_lo_prefit = (x_lo_prefit * x_prefit) / 1000.0
    #         x_prefit = x_prefit / 1000.0

    #         nround = 1 if x > 10 else 3

    #         ax.text(
    #             xRange[0] + 0.01 * (xRange[1] - xRange[0]),
    #             y + 0.2,
    #             rf"${round(x_prefit,nround)} \pm {round(abs(x_lo_prefit),nround)} \,\mathrm{{fb}}^{{-1}}$",
    #             va="center",
    #             ha="left",
    #             color="black",
    #         )
    #         ax.text(
    #             xRange[0] + 0.01 * (xRange[1] - xRange[0]),
    #             y - 0.2,
    #             rf"${round(x,nround)} \pm {round(abs(x_lo),nround)} \,\mathrm{{fb}}^{{-1}}$",
    #             va="center",
    #             ha="left",
    #             color="blue",
    #         )

    #     if chi2_info is not None:
    #         ax.text(
    #             0.01,
    #             0.99,
    #             rf"$\chi^2/ndf = {round(chi2_info[0],2)}/{chi2_info[1]}$"
    #             + "\n"
    #             + rf" $(p={chi2_info[2]}\%)$",
    #             va="top",
    #             ha="left",
    #             transform=ax.transAxes,
    #         )

    # if cms_decor is not None:
    #     hep.cms.label(label=cms_decor, loc=0, ax=ax, data=True)

    ax.set_xlabel("($\\hat{L} - L_0 ) / L_0$")
    ax.set_ylabel("")

    ax.legend(loc="upper right", ncol=1)

    # xmax = max(max(xx_hi), max(xx_hi_prefit))
    # xmin = -xmax

    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.set_xlim(xRange)
    ax.set_ylim(ymin, ymax + 1)

    yield_tables = [df]
    if df_blue is not None:
        yield_tables.append(df_blue)

    plot_tools.save_plot(outDir, "pulls_lumi", yield_tables=yield_tables)


def plot_scan(
    result, loss, minimizer, param, name="param", profile=True, limits=2.0, outDir="./"
):
    """
    plot likelihood scan
    """

    import zfit

    logger.info("Make likelihood scan for {0}".format(name))

    val = result.params[param]["value"]
    # err = result.params[param]["hesse"]["error"]

    # scan around +/- 2 sigma (prefit) interval
    xLo = val - limits
    xHi = val + limits

    x = np.linspace(xLo, xHi, num=100)
    y = []
    param.floating = False
    for val in x:
        param.set_value(val)
        if profile:
            minimizer.minimize(loss)
        y.append(loss.value())

    y = (np.array(y) - result.fmin) * 2

    param.floating = True
    zfit.param.set_values(loss.get_params(), result)

    # ymin = min(y)

    yargmin = np.argmin(y)
    # xmin = x[np.argmin(y)]
    # left and right 68% intervals
    xL1 = x[np.argmin(abs(y[:yargmin] - 1))]
    xR1 = x[yargmin + np.argmin(abs(y[yargmin:] - 1))]
    # left and right 95% intervals
    xL2 = x[np.argmin(abs(y[:yargmin] - 4))]
    xR2 = x[yargmin + np.argmin(abs(y[yargmin:] - 4))]

    plt.figure()

    if max(y) >= 1:
        plt.plot([xLo, xL1], [1, 1], linestyle="dashed", color="gray")
        plt.plot([xR1, xHi], [1, 1], linestyle="dashed", color="gray")
        plt.plot([xL1, xL1], [0, 1], linestyle="dashed", color="gray")
        plt.plot([xR1, xR1], [0, 1], linestyle="dashed", color="gray")

    if max(y) >= 4:
        plt.plot([xLo, xL2], [4, 4], linestyle="dashed", color="gray")
        plt.plot([xR2, xHi], [4, 4], linestyle="dashed", color="gray")
        plt.plot([xL2, xL2], [0, 4], linestyle="dashed", color="gray")
        plt.plot([xR2, xR2], [0, 4], linestyle="dashed", color="gray")

    plt.plot(x, y, color="red")
    plt.xlabel(name)
    plt.ylabel("-2 $\\Delta$ ln(L)")

    plt.xlim(xLo, xHi)
    plt.ylim(0, 5)

    plt.tight_layout()

    suffix = "_profiling" if profile else "_scan"
    name += suffix

    plot_tools.save_plot(outDir, name)
