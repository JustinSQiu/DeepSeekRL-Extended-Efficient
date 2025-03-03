import random
from sympy import Matrix

def generate_random_matrix(rows, cols, low, high):
    """Generates a random n x n matrix"""
    return [[random.randint(low, high) for _ in range(rows)] for _ in range(cols)]

def write_matrices_to_file(filename, matrices):
    """Writes matrices to a file in the required format."""
    with open(filename, 'w') as file:
        for matrix in matrices:
            file.write(f"{matrix}\n")

def main(num_matrices, rows=10, cols=10, low=-100, high=100):
    # Step 1: Generate random matrices
    matrices = [generate_random_matrix(random.randint(1, rows), random.randint(1, cols), low, high) for _ in range(num_matrices)]
    
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
    main(100)
