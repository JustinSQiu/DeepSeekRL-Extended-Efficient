import random
from sympy import Matrix

def generate_random_matrix(n):
    """Generates a random n x n matrix with values between -10 and 10."""
    return [[random.randint(-10, 10) for _ in range(n)] for _ in range(n)]

def write_matrices_to_file(filename, matrices):
    """Writes matrices to a file in the required format."""
    with open(filename, 'w') as file:
        for matrix in matrices:
            file.write(f"{matrix}\n")

def main(n, num_matrices):
    # Step 1: Generate random matrices
    matrices = [generate_random_matrix(n) for _ in range(num_matrices)]
    
    # Step 2: Write the matrices to inputs.txt
    write_matrices_to_file('data/inputs.txt', matrices)

    # Step 3: Calculate RREF for each matrix and write the results to outputs.txt
    rref_matrices = []
    for matrix in matrices:
        sym_matrix = Matrix(matrix)
        rref_matrix, _ = sym_matrix.rref()  # Get RREF of the matrix
        rref_matrices.append(str(rref_matrix.tolist()))

    # Step 4: Write RREF matrices to outputs.txt
    write_matrices_to_file('data/outputs.txt', rref_matrices)

    print("Matrices and their RREF have been written to 'inputs.txt' and 'outputs.txt'.")

if __name__ == "__main__":
    # Define matrix size and number of matrices
    n = 2  # Size of the matrix (n x n)
    num_matrices = 100  # Number of random matrices to generate
    
    main(n, num_matrices)
