# Steps to reproduce
1. Run setup.py
2. Download [real estate dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset/data)
3. Change csv file name to data.csv and put into data/real_estate
4. Download [HPI data](https://fred.stlouisfed.org/series/USSTHPI), [CPI data](https://fred.stlouisfed.org/series/CPIAUCNS), [Unemployment data](https://fred.stlouisfed.org/series/UNRATENSA) and [Real GDP data](https://fred.stlouisfed.org/series/A191RO1Q156NBEA) and put those files into data/macro
5. Rename Real GDP data file to REALGDP.csv
6. Run the whole data_preprocessing_and_eda notebook
7. Run pipeline.py
8. Run the whole evaluation notebook

If you would like to also include some geographical data (amenitites) then after step 6 adjust according to your needs and run the whole geo_data notebook (it is very slow).
