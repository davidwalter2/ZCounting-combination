import csv
import sys

import numpy as np

from utils import logging

logger = logging.child_logger(__name__)


def load_uncertainties(inputFile, years=None):
    logger.info(f"Load {inputFile}  uncertainties")

    values = None
    correlations = {}
    uncertainties = {}
    with open(inputFile, "r") as csv_file:
        reader = csv.reader(csv_file, skipinitialspace=True)
        for i, row in enumerate(reader):
            if len(row) == 0 or row[0][0] == "#":
                continue
            systName = row[0]
            if i == 0:
                if years is None:
                    # if no years are given, take all years from the file
                    years = row[2:]
                    select = [True for y in row[2:]]
                else:
                    new_years = []
                    for year in years:
                        if year not in row[2:]:
                            logger.error(f"year {year} not found in top line")
                            sys.exit(1)

                    # new array of years in correct order
                    new_years = []
                    for y in row[2:]:
                        if y in years:
                            new_years.append(y)
                    years = new_years

                    select = [y in years for y in row[2:]]
            elif i == 1:
                if systName not in ["Value", "Luminosity"]:
                    logger.error("expected first row to have vthe central alues")
                    sys.exit(1)
                values = np.array([float(x) for x in row[2:]])[select]
                values = {y: v for y, v in zip(years, values)}
                if len(years) != len(values):
                    logger.error(
                        "number of values specified doesn't match number of years"
                    )
                    sys.exit(1)
            else:
                if row[1] != "C" and row[1] != "U" and row[1][0] != "P":
                    logger.error(f"line {i}, correlation should be C, U, or P##")
                    sys.exit(1)
                correlations[systName] = row[1]
                uncertainties[systName] = np.array([float(x) / 100 for x in row[2:]])[
                    select
                ]
                if len(years) != len(uncertainties[systName]):
                    logger.error(
                        f"number of uncertainties for {systName} doesn't match number of years",
                    )
                    sys.exit(1)

    return correlations, uncertainties, values, years


def build_nuisances(years, correlations, uncertainties):
    logger.info("Build nuisance parameters")

    result = {}
    # nuisance_labels = {}
    for key, uncert in uncertainties.items():
        if all(u == 0 for u in uncert):
            logger.warning(f"All entries for uncertanty {key} found 0, skip")
            continue

        # nuisane_name = key[:]
        # for x,y in [("-",""), (" ",""), ("(","", ")")]
        #     nuisane_name = nuisane_name.replace(x,y)

        # check which years are correlated
        if correlations[key] == "U":
            # uncorrelated, make 1 nuisance parameter per year
            for i, year in enumerate(years):
                if uncert[i] == 0:
                    continue
                result[f"{key} [{year}]"] = {year: uncert[i]}
                # nuisance_labels[nuisance_name_i] = f"{key} [{year}]"

        elif correlations[key] == "C":
            # correlated, make 1 nuisance for all years
            result[key] = {
                year: uncert[i] for i, year in enumerate(years) if uncert[i] != 0
            }
            # nuisance_labels[nuisance_name] = key
        else:
            raise NotImplementedError(f"Unknown correlation type {correlations[key]}")
    return result


def load_nuisances(inputFile, years=None):
    corr, unc, val, years = load_uncertainties(inputFile, years)
    result = build_nuisances(years, corr, unc)

    _, covariance = build_covariance_matrix(
        years, corr, unc, np.array([v for v in val.values()])
    )

    return result, val, covariance


def build_covariance_matrix(years, correlations, uncertainties, values, prefix=""):
    logger.info("Build covariance matrix")

    # diagnoal matrix for uncorrelated uncertainties
    uncorr = np.zeros((len(years), len(years)))
    for i in range(len(years)):
        uncorr[i, i] = 1.0

    # covariance matrix to be filled
    covariance = np.zeros((len(years), len(years)))

    result = {}
    for key, uncert in uncertainties.items():
        # check which years are correlated
        if correlations[key] == "U":
            for i, y in enumerate(years):
                uname = prefix + y
                if uncert[i] == 0:  # skip if uncertainty is 0
                    continue
                if uname not in result:
                    result[uname] = {y: 0}
                result[uname][y] += uncert[i] ** 2

            covariance += uncorr * (uncert * values) ** 2

        elif correlations[key] == "C":
            thisSet = []
            for i in range(len(years)):
                if uncert[i] > 0:
                    thisSet.append(years[i])

            if len(thisSet) == 0:
                continue

            setName = prefix + "".join(thisSet)
            if setName not in result.keys():
                result[setName] = {y: 0 for y in thisSet}

            for i, y in enumerate(years):
                if uncert[i] > 0:
                    result[setName][y] += uncert[i] ** 2

            corr = np.ones((len(years), len(years)))
            for i in range(len(values)):
                for j in range(len(values)):
                    corr[i, j] = uncert[i] * values[i] * uncert[j] * values[j]

            covariance += corr
        else:
            raise NotImplementedError(f"Unknown correlation type {correlations[key]}")

    return result, covariance


def simplify_uncertainties(
    inputFile,
    years=None,
    ratio=False,
    force_correlated=False,
    force_uncorrelated=False,
    prefix="",
):
    """
    This script will combine the values for different years with uncertainties as specified in the input
    text file. The file should have the format:

    Description,               Corr,2015,2016,2017,2018
    Value,                     -,   4.21,40.99,49.79,67.86
    Length scale,              C,   0.5, 0.8, 0.3, 0.2
    Orbit drift,               C,   0.2, 0.1, 0.2, 0.1
    ...
    where the first line has the list of years
    the second line has the list of values
    and the all subsequent lines have, for each uncertainty, the correlation for that uncertainty and the value
    (in %) of that uncertainty for each year.
    The correlation should be 'C' for fully correlated uncertainties, 'U' for uncorrelated uncertainties, or for
    partially correlated, P## where ## is the percent correlation (e.g. 'P70' for a 70% correlated uncertainty).
    """

    correlations, uncertainties, values, years = load_uncertainties(inputFile, years)

    result, covariance = build_covariance_matrix(
        years,
        correlations,
        uncertainties,
        np.array([v for v in values.values()]),
        prefix,
    )

    for unc_name, unc_values in result.items():
        for unc_year, unc_value in unc_values.items():
            result[unc_name][unc_year] = np.sqrt(unc_value)

    return result, values, covariance
