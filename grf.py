# -*- coding: utf-8 -*-
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector


# import R's utility package
utils = rpackages.importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

packnames = ('grf', 'hte')

# Selectively install what needs to be install.
# names_to_install = [x for packnames if not rpackages.isinstalled(x)]
#if len(names_to_install) > 0:
#    utils.install_packages(StrVector(names_to_install))


# grf_ate_f = robjects.r['grf::average_treatment_effect']