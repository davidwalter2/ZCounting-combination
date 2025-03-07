import os

import hist
import numpy as np
import pandas as pd
import scipy
import uncertainties as unc
import zfit
from hist import Hist
from uncertainties import unumpy as unp

from utils import common, functions, logging, output_tools, parsing
from utils.plot_functions import plot_matrix
from utils.plot_functions_zfit import plot_pulls, plot_pulls_lumi, plot_scan

os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1."

parser = parsing.plot_parser()
parser.add_argument(
    "--eras",
    default=None,
    nargs="*",
    help="list of eras to be used in combination",
)
parser.add_argument(
    "--lumi",
    default=f"{common.dir_data}/uncertainty_tables/lumi_16171817H.csv",
    type=str,
    help="file containing uncertainties on luminosity",
)
parser.add_argument(
    "--zrate",
    default=f"{common.dir_data}/uncertainty_tables/zcounts.csv",
    type=str,
    help="file containing uncertainties on z rate",
)
parser.add_argument(
    "--saturated", default=False, action="store_true", help="Run saturated model"
)
parser.add_argument(
    "--chisqFit",
    default=False,
    action="store_true",
    help="Perform chi^2 fit instead of binned Poisson likelihood",
)
parser.add_argument(
    "--simplify",
    default=False,
    action="store_true",
    help="Simplify uncertainties by building covariance matrix",
)
parser.add_argument("--unblind", default=False, action="store_true", help="Fit on data")

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

outDir = output_tools.make_plot_dir(
    args.outpath, args.outfolder, eoscp=output_tools.is_eosuser_path(args.outpath)
)

# --- settings

# fiducial Z cross section at 13TeV in fb
xsec = 734
xsec_uncertainty = None  # no uncertainty on cross section - free floating parameter
# xsec_uncertainty = 0.03   # uncertainty on cross section - gaussian constraint

# ratio of fiducial cross sections at different center of mass energies (13.6 / 13TeV for 2022)
energy_extrapolation = {
    "2016": 1.0,
    "2016preVFP": 1.0,
    "2016postVFP": 1.0,
    "2017": 1.0,
    "2017H": 1.0,
    "2018": 1.0,
    "2022": 1.046,  # With NNLO + N3LL + NLO(EW)
}

# efficiencies to get from corrected to uncorrected number
z_efficiencies = {
    "2016": 0.880627183193766,
    "2016preVFP": 0.8729268470636643,
    "2016postVFP": 0.8893522069626418,
    "2017": 0.8932884309328363,
    "2017H": 0.9068543037179428,
    "2018": 0.9023466577682436,
}

# --- calculate initial values / prefit plots

if args.simplify:
    uncertainties_lumi, ref_lumi, covariance_lumi_prefit = (
        functions.simplify_uncertainties(args.lumi, args.eras, prefix="l")
    )
    uncertainties_z, z_yields, covariance_z_prefit = functions.simplify_uncertainties(
        args.zrate, args.eras, prefix="z"
    )
else:
    uncertainties_lumi, ref_lumi, covariance_lumi_prefit = functions.load_nuisances(
        args.lumi, args.eras
    )
    uncertainties_z, z_yields, covariance_z_prefit = functions.load_nuisances(
        args.zrate, args.eras
    )

# for BLUE input
# add statistical uncertainty on covariance for Z
diag_indices = np.diag_indices(min(covariance_z_prefit.shape))
covariance_z_prefit[diag_indices] += np.array([v for v in z_yields.values()])

lumi = unc.correlated_values(
    np.array([v for v in ref_lumi.values()]), covariance_lumi_prefit
)
nz = unc.correlated_values(
    np.array([v / z_efficiencies[k] for k, v in z_yields.items()]), covariance_z_prefit
)

# high pileup lumi estimations
lumi_z = np.array([nz[i] / nz[2] * lumi[3] for i in range(len(nz)) if i != 2])
lumi_l = np.array([lumi[l] for l in range(len(lumi) - 1)])
# covariances between these estimations

cov_matrix = unc.covariance_matrix([*lumi_l, *lumi_z])
std_devs = np.sqrt(np.diag(cov_matrix))
correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)

# df_blue = pd.DataFrame(
#     data={
#         "era": ["2016", "2017", "2018", "Sum"],
#         "postfit": [35933.854, 38454.986, 59141.564, 133530],
#         "postfit_uncertainty": [273.497, 266.411, 408.164, 906.661],
#         "prefit": [*[l.n for l in lumi_l], sum(lumi_l).n],
#         "prefit_uncertainty": [*[l.s for l in lumi_l], sum(lumi_l).s],
#         "prefit_z": [*[l.n for l in lumi_z], sum(lumi_z).n],
#         "prefit_z_uncertainty": [*[l.s for l in lumi_z], sum(lumi_z).s],
#     }
# )
df_blue = None

# uncertainties_z = {}

ref_lumi = {
    k: v
    for k, v in zip(
        ref_lumi.keys(),
        unc.correlated_values(
            np.array([v for v in ref_lumi.values()]), covariance_lumi_prefit
        ),
    )
}

z_yields = {
    k: v
    for k, v in zip(
        z_yields.keys(),
        unc.correlated_values(
            np.array([v for v in z_yields.values()]), covariance_z_prefit
        ),
    )
}

eras = [k for k in ref_lumi.keys()]

plot_matrix(
    covariance_lumi_prefit,
    labels=eras,
    name="lumi_prefit",
    matrix_type="covariance",
    outDir=outDir,
)

plot_matrix(
    covariance_lumi_prefit,
    labels=eras,
    name="lumi_prefit",
    matrix_type="correlation",
    outDir=outDir,
)

plot_matrix(
    covariance_z_prefit,
    labels=eras,
    name="zyield_prefit",
    matrix_type="covariance",
    outDir=outDir,
)


plot_matrix(
    covariance_z_prefit,
    labels=eras,
    name="zyield_prefit",
    matrix_type="correlation",
    outDir=outDir,
)

if not args.unblind:
    # Use asymov data
    z_yields = {
        key: value * xsec * z_efficiencies[key] * energy_extrapolation[key]
        for key, value in ref_lumi.items()
    }

# --- set up binned likelihood fit
nBins = len(eras)
binLo = 0
binHi = nBins

# set the Z counts
hZ = Hist(hist.axis.Regular(bins=nBins, start=binLo, stop=binHi, name="x"))

# set observed Z yields for data
hZ[:] = [int(z_yields[era].n) for era in eras]

# set the reference lumi counts
hists_ref = {}
for i, era in enumerate(eras):
    hist_ref = Hist(hist.axis.Regular(bins=nBins, start=binLo, stop=binHi, name="x"))

    # set lumi
    hist_ref[i] = ref_lumi[era].n

    hists_ref[era] = hist_ref

# obs_nobin = zfit.Space('x', (binLo, binHi))

binning = zfit.binned.RegularBinning(nBins, binLo, binHi, name="x")
obs = zfit.Space("x", binning=binning)

data = zfit.data.BinnedData.from_hist(hZ)

# make extended pdfs, each bin is scaled by a separate histogram

# cross section as a common normalization
rate_xsec = zfit.Parameter("r_xsec", 1.0, 0.8, 1.2)

# nuisance parameters on luminosity
nuisances_lumi = {
    key: zfit.Parameter(f"l{i}", 0, -5.0, 5.0, label=key)
    for i, key in enumerate(uncertainties_lumi.keys())
}

# nuisance parameters on Z yield
nuisances_z = {
    key: zfit.Parameter(f"z{i}", 0, -5.0, 5.0, label=key)
    for i, key in enumerate(uncertainties_z.keys())
}

# all nuisance parameters
nuisances = {**nuisances_lumi, **nuisances_z}

# put gaussian constraints on all nuisances
constraints = {
    key: zfit.constraint.GaussianConstraint(param, observation=0.0, uncertainty=1.0)
    for key, param in nuisances.items()
}
if args.saturated:
    # separate free floating cross section per bin
    constraints_saturated = constraints.copy()
    rate_xsec_saturated = [
        zfit.Parameter(f"r_xsec_{era}_saturated", 1.0, 0.8, 1.2) for era in eras
    ]

if xsec_uncertainty is not None:
    # put a gaussian constraint on the cross section
    constraints["r_xsec"] = zfit.constraint.GaussianConstraint(
        rate_xsec, observation=1.0, uncertainty=xsec_uncertainty
    )

# rearange dictionaries: for each era a list of uncertainties, nuisances
uncertainties_lumi_era = {}
nuisances_lumi_era = {}
uncertainties_z_era = {}
nuisances_z_era = {}
uncertainties_era = {}
nuisances_era = {}

for era in eras:
    uncertainties_lumi_era[era] = []
    nuisances_lumi_era[era] = []
    for key, uncertainty in uncertainties_lumi.items():
        if era in uncertainty.keys():
            uncertainties_lumi_era[era].append(uncertainty[era] + 1)
            nuisances_lumi_era[era].append(nuisances_lumi[key])

    uncertainties_z_era[era] = []
    nuisances_z_era[era] = []
    for key, uncertainty in uncertainties_z.items():
        if era in uncertainty.keys():
            uncertainties_z_era[era].append(uncertainty[era] + 1)
            nuisances_z_era[era].append(nuisances_z[key])

    uncertainties_era[era] = [*uncertainties_z_era[era], *uncertainties_lumi_era[era]]
    nuisances_era[era] = [*nuisances_z_era[era], *nuisances_lumi_era[era]]


def get_luminosity_function(era):
    # return function to calculate luminosity for a year

    # central value of luminosity
    lumi = ref_lumi[era].n

    # dictionary with uncertainties for the considered era
    def function(params):
        l = lumi
        for i, p in enumerate(params):
            l *= uncertainties_lumi_era[era][i] ** p
        return l

    return function


def get_nexp_function(era, unc_dict):
    # return function to calculate number of expected Z events

    z_eff = z_efficiencies[era]
    lumi = ref_lumi[era].n
    extrapolation = energy_extrapolation[era]

    nexp_0 = xsec * extrapolation * z_eff * lumi

    def function(params):
        nexp = nexp_0 * params[0]
        for i, p in enumerate(params[1:]):
            nexp *= unc_dict[era][i] ** p
        return nexp

    return function


models = {}
models_saturated = {}
# lumis = {}
# nexps = {}
for i, era in enumerate(eras):
    logger.info(f"create model for {era}")

    # lumi parameter including uncetainties
    l = zfit.ComposedParameter(
        f"lumi_{era}".format(era),
        get_luminosity_function(era),
        params=[nuisances_lumi_era[era]],
    )

    # Number of expected Z events, including uncertainties on cross section, acceptance, and efficiencies
    nexp = zfit.ComposedParameter(
        f"nexp_{era}".format(era),
        get_nexp_function(era, uncertainties_era),
        params=[
            rate_xsec,
            *nuisances_era[era],
        ],
    )

    # lumis[era] = l
    # nexps[era] = nexp
    models[era] = zfit.pdf.HistogramPDF(
        hists_ref[era], extended=nexp, name=f"PDF_Bin{era}"
    )

    if args.saturated:
        # make saturated model
        # Number of expected Z events, including uncertainties on cross section, acceptance, and efficiencies
        nexp_saturated = zfit.ComposedParameter(
            f"nexp_{era}".format(era),
            get_nexp_function(era, uncertainties_era),
            params=[
                rate_xsec_saturated[i],
                *nuisances_era[era],
            ],
        )

        models_saturated[era] = zfit.pdf.HistogramPDF(
            hists_ref[era], extended=nexp_saturated, name=f"PDF_Bin{era}_saturated"
        )

logger.info("Build composite model")
model = zfit.pdf.BinnedSumPDF([m for m in models.values()])

# if not args.unblind:
#     # azimov_hist = model.to_hist()
#     data = model.to_binneddata()

if args.chisqFit:
    nll = zfit.loss.ExtendedBinnedChi2
else:
    nll = zfit.loss.ExtendedBinnedNLL

loss = nll(
    model,
    data,
    constraints=[c for c in constraints.values()],
    options={"numhess": False},
)

logger.info("Minimize")
minimizer = zfit.minimize.ScipyTrustConstrV1(hessian="zfit")
result = minimizer.minimize(loss)
nll_value = loss.value().numpy()

logger.info(f"status: {result.valid}")
logger.info(f"loss = {nll_value}")

try:
    hessval = result.loss.hessian(list(result.params)).numpy()
    cov = np.linalg.inv(hessval)
    eigvals = np.linalg.eigvalsh(hessval)
    covstatus = eigvals[0] > 0.0
    logger.info(f"Eigen values: {eigvals}")
except Exception as e:
    logger.info(f"An error occurred: {e}")
    cov = None
    covstatus = False

logger.info(f"covariance status: {covstatus}")

if args.saturated:
    logger.info("Build composite saturated model")
    model_saturated = zfit.pdf.BinnedSumPDF([m for m in models_saturated.values()])
    loss_saturated = zfit.loss.ExtendedBinnedNLL(
        model_saturated,
        data,
        constraints=[c for c in constraints_saturated.values()],
        options={"numhess": False},
    )
    logger.info("Minimize")
    result_saturated = minimizer.minimize(loss_saturated)
    nll_saturated_value = loss_saturated.value().numpy()

    logger.info(f"status: {result_saturated.valid}")
    logger.info(f"loss = {nll_saturated_value}")

    chi2_saturated = 2 * (nll_value - nll_saturated_value)
    ndf = len(rate_xsec_saturated) - (xsec_uncertainty is None)
    p_val = int(round(scipy.stats.chi2.sf(chi2_saturated, ndf) * 100))

    chi2_info = [chi2_saturated, ndf, p_val]

    logger.info(rf"2*(nll-nll_sat.)/ndf = {chi2_saturated}/{ndf} (p={p_val}%)")
else:
    chi2_info = None

# --- error propagation to get correlated uncertainty on parameters
correlated_values = unc.correlated_values(
    [result.params[p]["value"] for p in result.params], cov
)

for p, v in zip(result.params, correlated_values):
    result.params[p]["correlated_value"] = v

lumi_function = [get_luminosity_function(era) for era in eras]
lumi_values = [
    lumi_function[i]([result.params[iv]["correlated_value"] for iv in v])
    for i, (e, v) in enumerate(nuisances_lumi_era.items())
]

lumi_prefit = [ref_lumi[era] for era in eras]
lumi_prefit.append(sum([ref_lumi[era] for era in eras if era != "2017H"]))

eras.append("Sum")
lumi_values.append(
    sum(
        [
            lumi_function[i]([result.params[iv]["correlated_value"] for iv in v])
            for i, (e, v) in enumerate(nuisances_lumi_era.items())
            if e != "2017H"
        ]
    )
)

df_lumi = pd.DataFrame(
    data={
        "era": eras,
        "postfit": unp.nominal_values(lumi_values),
        "postfit_uncertainty": unp.std_devs(lumi_values),
        "prefit": [x.n for x in lumi_prefit],
        "prefit_uncertainty": [x.s for x in lumi_prefit],
    }
)

df_lumi["prefit_relative_uncertainty"] = (
    df_lumi["prefit_uncertainty"] / df_lumi["prefit"]
)
df_lumi["postfit_relative_uncertainty"] = (
    df_lumi["postfit_uncertainty"] / df_lumi["postfit"]
)

logger.info(df_lumi)

corr_matrix_lumi = unc.correlation_matrix(lumi_values)
cov_matrix_lumi = (
    np.array(unc.covariance_matrix(lumi_values)) / 1000000.0
)  # covariance matrix in fb

# logger.info(corr_matrix_lumi)
# logger.info(cov_matrix_lumi)

all_params = [
    rate_xsec,
] + [n for n in nuisances.values()]

# --- plotting

plot_matrix(
    corr_matrix_lumi,
    labels=eras,
    name="lumi_postfit",
    matrix_type="correlation",
    outDir=outDir,
)
plot_matrix(
    cov_matrix_lumi,
    labels=eras,
    name="lumi_postfit",
    matrix_type="covariance",
    outDir=outDir,
)

# pull plot for uncertainties related to PHYSICS lumi
params_l = [result.params[p] for p in result.params if p.name.startswith("l")]
names_l = np.array([p.label for p in result.params if p.name.startswith("l")])

plot_pulls(
    params_l,
    names_l,
    outDir=outDir,
    cms_decor=args.cmsDecor,
    postfix="uncertainty_lumi",
)

# pll plot for uncertainties related to NZ
params_z = [
    result.params[p]
    for p in result.params
    if p.name.startswith("z") or p.name == "r_xsec"
]
names_z = np.array(
    [p.label for p in result.params if p.name.startswith("z") or p.name == "r_xsec"]
)

plot_pulls(
    params_z, names_z, outDir=outDir, cms_decor=args.cmsDecor, postfix="uncertainty_z"
)


# for i in range(int(len(result.params) / 20) + 1):
#     if len(result.params) < (i + 1) * 20:
#         idx_hi = None
#     else:
#         idx_hi = (i + 1) * 20

#     idx_lo = i * 20 if i != 0 else None

#     plot_pulls(
#         result, idx_lo=idx_lo, idx_hi=idx_hi, outDir=outDir, cms_decor=args.cmsDecor
#     )

plot_pulls_lumi(
    df_lumi,
    df_blue=df_blue,
    chi2_info=chi2_info,
    outDir=outDir,
    cms_decor=args.cmsDecor,
)

# plot_scan(result, loss, minimizer, rate_xsec, "r_xsec", limits=0.03, profile=False, outDir=outDir)
# plot_scan(result, loss, minimizer, rate_xsec, "r_xsec", limits=0.03, outDir=outDir)

# for era, p in lumis.items():
#     plot_scan(result, p, "l_"+era, limits=0.1)

if output_tools.is_eosuser_path(args.outpath):
    output_tools.copy_to_eos(outDir, args.outpath, args.outfolder)

exit()

for n, p in nuisances.items():
    # plot_scan(result, loss, minimizer, p, "n_"+n, profile=False, outDir=outDir)
    plot_scan(result, loss, minimizer, p, "n_" + n, outDir=outDir)
