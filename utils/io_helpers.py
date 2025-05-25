def print_dataframe(df, num_rows=20):
    print(f"\nFirst {num_rows} rows of the dataset:")
    print(df.head(num_rows))

def save_dataframe_to_txt(df, filename="data_output.txt"):
    with open(filename, "w") as f:
        f.write(df.to_string(index=True))
