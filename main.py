# Import required libraries for plotting and numerical operations
import matplotlib.pyplot as plt
import numpy as np

# Define constants
VECTOR_COUNT = 1000
THRESHOLD = 0.05
FILE = "out/ex1.png"

# Generate arrays with specified dimensions, number, and distribution.
def generate_arrays(dimension=10000, n=VECTOR_COUNT, distribution=np.random.normal):
    # Use list comprehension to generate 'n' vectors each with 'dimension' size
    return [distribution(0, 1. / np.sqrt(dimension), dimension) for i in range(n)]


# Calculate dot products between all vectors in the input list.
def dot_products(arrays: list):
    __products = []
    for i in range(VECTOR_COUNT):
        __products.append([])  # Start a new list for current vector
        for j in range(VECTOR_COUNT):
            if i == j:
                # Dot product of a vector with itself is set to 0 (to ignore)
                __products[i].append(0)
                continue
            # Compute the dot product and store it in the matrix
            __products[i].append(np.dot(arrays[i], arrays[j]))
    return __products


# Flatten the matrix to get a list of percentages
# of products in each vector that are below the threshold.
def flatten(matrix: list) -> list:
    flat = []
    for vector in matrix:
        v = np.array(vector)  # Convert list to numpy array for vectorized operations
        x = (np.abs(v) <= THRESHOLD).size / v.size
        flat.append(x)
    return flat


# Convert 2D matrix to 1D list.
def linearize(matrix: list) -> list:
    flat = []
    for vector in matrix:
        flat += vector  # Extend the flat list with current vector
    return flat


# Utility function to plot a histogram.
def plot(series: list, t, x, y, index):
    plt.subplot(index)
    plt.hist(series, bins=50, alpha=0.5)  # Plot histogram
    plt.title(t)
    plt.xlabel(x)
    plt.ylabel(y)


# Display the plots with a super title.
def show(suptitle):
    plt.suptitle(suptitle)
    plt.tight_layout()  # Adjust subplot layout
    plt.show()


# Save image to file.
def save(filename):
    plt.savefig(filename, bbox_inches="tight")


# Main execution starts here
if __name__ == '__main__':
    # Compute dot products of the vectors
    products = dot_products(generate_arrays())
    print(f"[+] Number of product below {THRESHOLD}: {np.sum((np.array(products) <= THRESHOLD))}")

    # Plot distribution of dot products
    t = f"Distribution of products"
    x = "Value"
    y = "Frequency"
    plot(linearize(products), t, x, y, 211)

    # Plot likelihood of dot products below threshold
    t = f"Likelihood of V[i] s.t. V[i] dot V[j] <= {THRESHOLD} for all i,j in [0,n], (j!=i)"
    x = "Value"
    y = "Frequency"
    plot(flatten(products), t, x, y, 212)

    # Save the plots
    save(FILE)
    # Display the plots
    show(f"Dot products distribution and N vectors < {THRESHOLD} distribution")
