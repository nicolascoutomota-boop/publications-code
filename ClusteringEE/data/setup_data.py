#####################################################################################################################
# Processing the publicly-available data into the data set provided here
#   This script does some data cleaning and organization steps for the analysis data set. The original copy of the
#   data set can be obtained at https://dataverse.unc.edu/dataset.xhtml?persistentId=doi:10.15139/S3/ZK60QH
#
# Paul Zivich (2026/06/17)
#####################################################################################################################

# Loading packages for data cleaning
import numpy as np
import pandas as pd

# Loading the data from the SAS file
d = pd.read_sas("deidc1june2017.sas7bdat")
cols_to_keep = ['c5',     # Site / venue ID (clustering)
                'c4',     # Country
                'c12',    # age
                'c14',    # sex
                'c15',    # education
                'c16',    # employed
                'c36',    # Age at first sex
                'c57',    # STI symptoms
                'c62',    # Ever had an HIV test
                'c117b',  # HIV test
                'c11a',   # Agreed to receive HIV test
                'c63b',   # Ever positive for HIV
                ]

# Subsetting to the columns we are interested in
dc = d[cols_to_keep].copy()

# Variable processing
dc['pid'] = dc.index + 1 + 900000
dc['venue_id'], uids = pd.factorize(d['sitespotid'])
dc['venue_id'] = dc['venue_id'] + 7000 + 1

dc['country'] = dc['c4']

dc['age'] = dc['c12']

dc['female'] = np.where(dc['c14'] == 2, 1, 0)
dc['female'] = np.where(dc['c14'].isna(), np.nan, dc['female'])

dc['educ'] = np.where(dc['c15'] == 999, np.nan, dc['c15'])

dc['ever_sex'] = np.where(dc['c16'] == 777, 0, 1)

dc['sexual_debut'] = np.where(dc['c16'] == 777, np.nan, dc['c36'])

dc['sti_symptoms'] = np.where(dc['c57'] == 999, np.nan, dc['c57'])

dc['ever_test'] = np.where(dc['c62'].isin([888, 999]), np.nan, dc['c62'])

dc['hiv_test'] = dc['c117b']
dc['hiv_test'] = np.where(dc['c117b'] == 3, np.nan, dc['hiv_test'])
dc['hiv_test'] = np.where(dc['c11a'] == 2, np.nan, dc['hiv_test'])
dc['hiv_test'] = 2 - dc['hiv_test']

# Subsetting to the processed data and saving as a CSV
cols_subset = ['pid', 'venue_id', 'female', 'educ', 'age', 'country',
               'ever_sex', 'sexual_debut', 'sti_symptoms', 'ever_test', 'hiv_test']
dc = dc[cols_subset].copy()
dc = dc.dropna(subset=['educ', 'sti_symptoms', 'ever_test', 'country'])
dc.to_csv("cbihs.csv", index=False)

# END
