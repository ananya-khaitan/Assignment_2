library(haven)
dta_path <- 'C:/Users/ANANYA/Desktop/BITS/3-2/RC/Assignment 2/4-1/HarmonizedGEM_2015_22Final/HarmonizedGEM_2015_22Final.dta'
meta <- read_dta(dta_path, n_max = 5)
cols <- colnames(meta)
cat('TEA columns: ', cols[grep('TEA', cols, ignore.case=TRUE)], '\n')
cat('OME columns: ', cols[grep('OME', cols, ignore.case=TRUE)], '\n')
cat('NME columns: ', cols[grep('NME', cols, ignore.case=TRUE)], '\n')
cat('Country columns: ', cols[grep('country', cols, ignore.case=TRUE)], '\n')