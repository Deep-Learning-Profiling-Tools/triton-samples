import triton.profiler.viewer as viewer

"""
Problem description:
1. Generate triton-{0...N}.hatchet
2. Flatten triton-{0...N}.hatchet to have depth=1
3. Sum triton-{0...N}.hatchet into triton.hatchet and divide by N
4. Compare profile.hatchet against golden.hatchet, and log kernels_diff for which profile[kernel][time] > atol + golden[kernel][time] * rtol
"""


def get_grouped_metrics(filename):
    with open(filename, "r") as f:
        gf, inclusive_metrics, exclusive_metrics, device_info = viewer.get_raw_metrics(
            f
        )
    return gf.dataframe.groupby("name").sum()


def compare_profiles(df1, df2, golden, atol):
    df1["avg_time (ns)"] = (df1["time (ns)"] + df2["time (ns)"]) / 2
    outlier_indices = df1["avg_time (ns)"] > (golden["time (ns)"] + atol)
    return df1[outlier_indices], df2[outlier_indices], golden[outlier_indices]


# Make sure triton-1.hatchet and triton-2.hatchet are in the current directory
df1 = get_grouped_metrics("triton-1.hatchet")
df2 = get_grouped_metrics("triton-2.hatchet")
golden = get_grouped_metrics("golden.hatchet")

atol = 100.0
outliers_df1, outliers_df2, outliers_golden = compare_profiles(df1, df2, golden, atol)

print("Golden:", outliers_golden["time (ns)"])
print("Outlier:", outliers_df1["avg_time (ns)"])
