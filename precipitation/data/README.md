# Data runs

1. Version: subset of variables 
2. Version: include 2 upstream predictors (geodiff and vvmean)
3. Version: include 3 more local predictors (vimd, stream and tendency)

Conclusion: Does subset of variables (remove not so important features) improve performance? (i.e. cin, shear, tendency...)

4. Version: replace 2 last upstream predictors by local variables (geodiff, vvmean)
5. Version: replace all upstream predictors by local variables (corr1, corr2, corr3, geodiff, vvmean)
6. Version: completely remove corr1, corr2 and corr3

Conclusion: Do we need "correlation" preprocessing in CNN context?. Do we need time lagging of precip at all? 

