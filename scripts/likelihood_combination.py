import os

import hist
import numpy as np
import pandas as pd
import uncertainties as unc
import zfit
from hist import Hist
from plot_utils_zfit import plot_matrix, plot_pulls, plot_pulls_lumi, plot_scan
from uncertainties import unumpy as unp

from common import common, parsing
from common.utils import simplify_uncertainties

os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1."

parser = parsing.base_parser()
parser.add_argument(
    "--eras",
    default=["2017", "2017H", "2018"],
    nargs="+",
    help="list of eras to be used in combination",
)
parser.add_argument(
    "--lumi",
    default=f"{common.dir_reources}/combination_tables/lumi_16171817H.txt",
    type=str,
    help="file containing uncertainties on luminosity",
)
parser.add_argument(
    "--zrate",
    default=f"{common.dir_reources}/combination_tables/zcounts_ratio.txt",
    type=str,
    help="file containing uncertainties on z rate",
)
parser.add_argument("--unblind", default=False, action="store_true", help="Fit on data")
parser.add_argument("-o", "--output", default="Test", type=str, help="give output dir")
args = parser.parse_args()

eras = args.eras

outDir = args.output
if not os.path.isdir(outDir):
    os.mkdir(outDir)

# --- settings

# fiducial Z cross section at 13TeV
xsec = 734
xsec_uncertainty = None  # no uncertainty on cross section - free floating parameter
# xsec_uncertainty = 0.03   # uncertainty on cross section - gaussian constraint

# ratio of fiducial cross sections at different center of mass energies (13.6 / 13TeV for 2022)
energy_extrapolation = {
    "2016preVFP": 1.0,
    "2016postVFP": 1.0,
    "2017": 1.0,
    "2017H": 1.0,
    "2018": 1.0,
    "2022": 1.046,  # With NNLO + N3LL + NLO(EW)
}

# statistical uncertainties of Z yields, right now hard coded TODO
z_yields_stat = {
    "2016preVFP": 0.00030886129248932546,
    "2016postVFP": 0.0003240928650176417,
    "2017": 0.00021328236230086072,
    "2017H": 0.002884547355729706,
    "2018": 0.0001702104409933952,
    "2022": 0.0002,  # Just a dummy number for now
}

# --- calculate initial values / prefit plots

uncertainties_lumi, ref_lumi = simplify_uncertainties(args.lumi, eras, prefix="l")
uncertainties_z, z_yields = simplify_uncertainties(args.zrate, eras, prefix="z")

covariance_lumi_prefit = unc.correlation_matrix(
    [ref_lumi[e] for e in eras]
    + [
        sum(ref_lumi.values()),
    ]
)
plot_matrix(
    covariance_lumi_prefit,
    labels=eras
    + [
        "Sum",
    ],
    name="lumi_prefit",
    matrix_type="correlation",
    outDir=outDir,
)

covariance_z_prefit = unc.correlation_matrix(
    [z_yields[e] for e in eras]
    + [
        sum(z_yields.values()),
    ]
)
plot_matrix(
    covariance_z_prefit,
    labels=eras
    + [
        "Sum",
    ],
    name="zyield_prefit",
    matrix_type="correlation",
    outDir=outDir,
)

if not args.unblind:
    # Use asymov data
    z_yields = {
        key: value * xsec * energy_extrapolation[key] for key, value in ref_lumi.items()
    }


# --- set up binned likelihood fit
nBins = len(eras)
binLo = 0
binHi = nBins

# set the Z counts
hZ = Hist(hist.axis.Regular(bins=nBins, start=binLo, stop=binHi, name="x"))

# set yields and variance (statistical uncertainty)
z_yields_uncorrected = {
    era: (1.0 / (z_yields_stat[era]) ** 2) for era in eras
}  # the uncorrected z yields is rederived from the statistical uncertainty
z_weights = {
    era: z_yields[era].n / z_yields_uncorrected[era] for era in eras
}  # the per event weights to get from uncorrected to corrected yield
# hZ[:] = [(z_yields[era], z_yields_uncorrected[era] * z_weights[era]**2) for era in eras]      # the variance is given by square of weights times uncorrected yields

# set uncorrected yields for data
hZ[:] = [int(z_yields_uncorrected[era]) for era in eras]

# efficiencies to get from corrected to uncorrected number
z_efficiencies = {era: z_yields_uncorrected[era] / z_yields[era].n for era in eras}


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
rate_xsec = zfit.Parameter("r_xsec", 0, -1.0, 1.0)

# nuisance parameters on luminosity
nuisances_lumi = {
    key: zfit.Parameter("n_{0}".format(key), 0, -5.0, 5.0)
    for key in uncertainties_lumi.keys()
}

# nuisance parameters on Z yield
nuisances_z = {
    key: zfit.Parameter("n_{0}".format(key), 0, -5.0, 5.0)
    for key in uncertainties_z.keys()
}

# all nuisance parameters
nuisances = {**nuisances_lumi, **nuisances_z}

# put gaussian constraints on all nuisances
constraints = {
    key: zfit.constraint.GaussianConstraint(param, observation=0.0, uncertainty=1.0)
    for key, param in nuisances.items()
}

if xsec_uncertainty is not None:
    # put a gaussian constraint on the cross section
    constraints["r_xsec"] = zfit.constraint.GaussianConstraint(
        rate_xsec, observation=0.0, uncertainty=xsec_uncertainty
    )


# rearange dictionaries: for each era a list of uncertainties, nuisances
uncertainties_lumi_era = {}
nuisances_lumi_era = {}
uncertainties_z_era = {}
nuisances_z_era = {}
for era in eras:
    uncertainties_lumi_era[era] = []
    nuisances_lumi_era[era] = []
    for key, uncertainty in uncertainties_lumi.items():
        if era in uncertainty.keys():
            uncertainties_lumi_era[era].append(uncertainty[era] / 100.0 + 1)
            nuisances_lumi_era[era].append(nuisances_lumi[key])

    uncertainties_z_era[era] = []
    nuisances_z_era[era] = []
    for key, uncertainty in uncertainties_z.items():
        if era in uncertainty.keys():
            uncertainties_z_era[era].append(uncertainty[era] / 100.0 + 1)
            nuisances_z_era[era].append(nuisances_z[key])


def get_luminosity_function(era):
    # return function to calculate luminosity for a year

    # central value of luminosity
    central = ref_lumi[era].n

    # dictionary with uncertainties for the considered era
    def function(params):
        l = central
        for i, p in enumerate(params):
            l *= uncertainties_lumi_era[era][i] ** p
        return l

    return function


def get_scale_function(era):

    z_eff = z_efficiencies[era]
    extrapolation = energy_extrapolation[era]

    def function(rate_xsec, lumi, params):  # , parameters=[]):
        s = xsec * extrapolation * z_eff * (1 + rate_xsec) * lumi
        # apply nuisance parameters
        # for p in parameters:
        #     s *= p
        for i, p in enumerate(params):
            s *= uncertainties_z_era[era][i] ** p
        return s

    return function


models = {}
lumis = {}
scales = {}
p_lumis = {}
for era in eras:
    print("create model for {0}".format(era))

    # p_lumi = zfit.Parameter('l_{0}'.format(era), 1, 0.5, 1.5, floating=True)

    # lumi parameter including uncetainties
    l = zfit.ComposedParameter(
        "lumi_{0}".format(era),
        get_luminosity_function(era),
        # params=[p_lumi, nuisances_lumi_era[era]]
        params=[nuisances_lumi_era[era]],
    )

    # absolute scale including uncertainties on cross section, acceptance, and efficiencies
    s = zfit.ComposedParameter(
        "scale_{0}".format(era),
        get_scale_function(era),
        params=[rate_xsec, l, nuisances_z_era[era]],
    )

    m = zfit.pdf.HistogramPDF(hists_ref[era], extended=s, name="PDF_Bin{0}".format(era))

    lumis[era] = l
    models[era] = m
    scales[era] = s
    # p_lumis[era] = p_lumi


# build composite model
model = zfit.pdf.BinnedSumPDF([m for m in models.values()])

# if not args.unblind:
#     # azimov_hist = model.to_hist()
#     data = model.to_binneddata()

loss = zfit.loss.ExtendedBinnedNLL(
    model,
    data,
    constraints=[c for c in constraints.values()],
    options={"numhess": False},
)

minimizer = zfit.minimize.ScipyTrustConstrV1(hessian="zfit")
result = minimizer.minimize(loss)
status = result.valid

print(f"status: {status}")

try:
    hessval = result.loss.hessian(list(result.params)).numpy()
    cov = np.linalg.inv(hessval)
    eigvals = np.linalg.eigvalsh(hessval)
    covstatus = eigvals[0] > 0.0
    print("eigvals", eigvals)
except Exception as e:
    print(f"An error occurred: {e}")
    cov = None
    covstatus = False

print(f"covariance status: {covstatus}")


# --- error propagation to get correlated uncertainty on parameters
correlated_values = unc.correlated_values(
    [result.params[p]["value"] for p in result.params], cov
)

for p, v in zip(result.params, correlated_values):
    result.params[p]["correlated_value"] = v

lumi_function = [get_luminosity_function(era) for era in eras]
lumi_values = [
    lumi_function[i]([result.params[iv]["correlated_value"] for iv in v])
    for i, v in enumerate(nuisances_lumi_era.values())
]

lumi_prefit = [ref_lumi[era] for era in eras]
lumi_prefit.append(sum(lumi_prefit))

eras.append("Sum")
lumi_values.append(sum(lumi_values))

df_lumi = pd.DataFrame(
    data={
        "era": eras,
        "value": unp.nominal_values(lumi_values),
        "hesse": unp.std_devs(lumi_values),
    }
)

df_lumi["prefit"] = lumi_prefit

df_lumi["relative_hesse"] = df_lumi["hesse"] / df_lumi["value"]

print(df_lumi)

corr_matrix_lumi = unc.correlation_matrix(lumi_values)
cov_matrix_lumi = (
    np.array(unc.covariance_matrix(lumi_values)) / 1000000.0
)  # covariance matrix in fb

# print(corr_matrix_lumi)
# print(cov_matrix_lumi)

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

plot_pulls(result, outDir=outDir)

plot_pulls_lumi(df_lumi, outDir=outDir)

exit()

# plot_scan(result, loss, minimizer, rate_xsec, "r_xsec", limits=0.03, profile=False, outDir=outDir)
plot_scan(result, loss, minimizer, rate_xsec, "r_xsec", limits=0.03, outDir=outDir)

# for era, p in p_lumis.items():
#     plot_scan(result, p, "l_"+era, limits=0.1)

for n, p in nuisances.items():
    # plot_scan(result, loss, minimizer, p, "n_"+n, profile=False, outDir=outDir)
    plot_scan(result, loss, minimizer, p, "n_" + n, outDir=outDir)
