
###############################################################################
### MERGE
# http://pandas.pydata.org/pandas-docs/stable/merging.html
rng = pd.date_range('2000-01-01', periods=6)
df1 = pd.DataFrame(np.random.randn(6, 3), index=rng, columns=['A', 'B', 'C'])
df2 = df1.copy()
# Append two data frames with overlapping index (emulate R rbind)
df = df1.append(df2, ignore_index=True); df

## Self-join of a data frame
df = pd.DataFrame(data={'Area' : ['A'] * 5 + ['C'] * 2,
                        'Bins' : [110] * 2 + [160] * 3 + [40] * 2,
                        'Test_0' : [0, 1, 0, 1, 2, 0, 1],
                        'Data' : np.random.randn(7)}); df
df['Test_1'] = df['Test_0'] - 1
pd.merge(df, df,
         left_on=['Bins', 'Area','Test_0'],
         right_on=['Bins', 'Area','Test_1'],
         suffixes=('_L','_R'))

# Concatenating pandas objects
df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)


###############################################################################
### JOIN
# http://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html#compare-with-sql-join
# pandas emulates SQL merge behavior
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

#
df = pd.DataFrame({
    'float_col' : [0.1, 0.2, 0.2, 10.1, np.NaN],
    'int_col'   : [1, 2, 6, 8, -1],
    'str_col'   : ['a', 'b', None, 'c', 'a']
})

other = pd.DataFrame({'str_col' : ['a','b'],
                      'some_val' : [1, 2]})

pd.merge(df, other, on='str_col', how="inner")
pd.merge(df, other, on='str_col', how="outer")
pd.merge(other, df, on='str_col', how="outer")
pd.merge(df, other, on='str_col', how="left")
pd.merge(df, other, on='str_col', how="right")


### APPEND
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
s = df.iloc[3]
df.append(s, ignore_index=True)

###############################################################################