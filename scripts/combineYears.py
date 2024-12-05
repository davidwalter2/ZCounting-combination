#!/usr/bin/env python

import argparse

from common import utils

# This script will combine the luminosities for different years with uncertainties as specified in the input
# text file. The file should have the format:
#
# Description,               Corr,2015,2016,2017,2018
# Luminosity,                -,   4.21,40.99,49.79,67.86
# Length scale,              C,   0.5, 0.8, 0.3, 0.2
# Orbit drift,               C,   0.2, 0.1, 0.2, 0.1
# ...
# where the first line has the list of years
# the second line has the list of luminosities
# and the all subsequent lines have, for each uncertainty, the correlation for that uncertainty and the value
# (in %) of that uncertainty for each year.
# The correlation should be 'C' for fully correlated uncertainties, 'U' for uncorrelated uncertainties, or for
# partially correlated, P## where ## is the percent correlation (e.g. 'P70' for a 70% correlated uncertainty).
#
# Command-line arguments:
# -y YEARS (e.g. -y 2016 2017 2018): add up only these selected years
# -c: force all uncertainties to be treated as correlated
# -u: force all uncertainties to be treated as uncorrelated

parser = argparse.ArgumentParser()
parser.add_argument("inputFile", help="Input file")
parser.add_argument(
    "-y", "--years", nargs="*", type=str, help="List of years to use in result"
)
parser.add_argument(
    "-r",
    "--ratio",
    action="store_true",
    help="Combining two different years at two energies",
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-c",
    "--force-correlated",
    action="store_true",
    help="Treat all systematics as correlated",
)
group.add_argument(
    "-u",
    "--force-uncorrelated",
    action="store_true",
    help="Treat all systematics as uncorrelated",
)
args = parser.parse_args()

result, values = utils.simplify_uncertainties(
    args.inputFile,
    args.years,
    args.ratio,
    args.force_correlated,
    args.force_uncorrelated,
)

total = sum([v for v in values.values()])
print("")
print(rf"Total luminosity is {total} (uncertainty of {round(total.s/total.n*100,2)}%)")

print("")
print("Luminosity and uncertainty per year is")
for y, v in values.items():
    print(f"{y}: {v} (uncertainty of {round(v.s/v.n*100,2)}%)")

print("")
print("Simplified correlation scheme")

for years, yearsSet in result.items():
    print(years)
    for year, uncertainty in yearsSet.items():
        print(f"{year}: {uncertainty}")
