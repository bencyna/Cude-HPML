import matplotlib.pyplot as plt

K_values = [1, 5, 10, 50, 100]

# CPU times 
cpu_times = [
    3.57906,   
    18.1666,   
    34.7719,   
    176.044,   
    343.256,   
]

# without Unified Memory
gpu_no_um_s1 = [215.192, 681.789, 1222.77, 5833.86, 11644.6]
gpu_no_um_s2 = [4.17143, 16.066, 31.6915, 156.87, 316.094]
gpu_no_um_s3 = [1.95815, 9.21255, 17.6736, 88.1058, 175.326]

# with unified memory
gpu_um_s1 = [172.304, 601.41, 1022.04, 4903.72, 9833.47]
gpu_um_s2 = [1.82115, 7.71544, 15.2431, 74.4439, 148.675]
gpu_um_s3 = [0.206334, 0.564624, 1.0685, 4.84766, 9.61244]

# plot no unified mem
plt.figure()
plt.plot(K_values, cpu_times, marker="o", label="CPU")

plt.plot(K_values, gpu_no_um_s1, marker="o", label="GPU (no UM) - Scenario 1")
plt.plot(K_values, gpu_no_um_s2, marker="o", label="GPU (no UM) - Scenario 2")
plt.plot(K_values, gpu_no_um_s3, marker="o", label="GPU (no UM) - Scenario 3")

plt.yscale("log")
plt.xlabel("K (millions of elements)")
plt.ylabel("Execution time (ms)")
plt.title("Step 2: GPU Without Unified Memory (with CPU baseline)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("q4_without_unified.png", dpi=150)

# plot with UM
plt.figure()
plt.plot(K_values, cpu_times, marker="o", label="CPU")

plt.plot(K_values, gpu_um_s1, marker="o", label="GPU (UM) - Scenario 1")
plt.plot(K_values, gpu_um_s2, marker="o", label="GPU (UM) - Scenario 2")
plt.plot(K_values, gpu_um_s3, marker="o", label="GPU (UM) - Scenario 3")

plt.yscale("log")
plt.xlabel("K (millions of elements)")
plt.ylabel("Execution time (ms)")
plt.title("Step 3: GPU With Unified Memory (with CPU baseline)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("q4_with_unified.png", dpi=150)

plt.show()
