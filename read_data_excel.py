import pandas as pd

import sys

excelFile = pd.read_excel(str(sys.argv[1]))

print('read excel file!')

excelFile_name=str(sys.argv[1])
excel_names = excelFile_name.split('.')

##convert to csv to manage better the column of data using Dataframe, csv is faster for reading

excelFile.to_csv (excel_names[0]+'.csv', index = None, header=True)

print('.csv file converted!')
