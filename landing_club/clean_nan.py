
# Remove records with nan
acc_df = acc_raw_df.loc[:, col_names]
#             .query('loan_amnt != 0')

acc_df = acc_df.dropna()
print(acc_df.sample(10))
acc_df.describe()
acc_df.int_rate.describe()
