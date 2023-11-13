import pandas as pd

# Create two sample dataframes
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                    'value': [1, 2, 3, 4]})

df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'],
                    'value': [5, 6, 7, 8]})

# Perform an inner join on the 'key' column
result_inner = pd.merge(df1, df2, on='key')

# Perform a left join on the 'key' column
result_left = pd.merge(df1, df2, on='key', how='left')

# Perform a right join on the 'key' column
result_right = pd.merge(df1, df2, on='key', how='right')

# Perform an outer join on the 'key' column
result_outer = pd.merge(df1, df2, on='key', how='outer')

print("Inner Join:\n", result_inner)
print("\nLeft Join:\n", result_left)
print("\nRight Join:\n", result_right)
print("\nOuter Join:\n", result_outer)
