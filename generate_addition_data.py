import random

def generate_addition_problem(low, high):
    """Generates a random addition problem (a, b) with a solution a + b."""
    a = random.randint(low, high)
    b = random.randint(low, high)
    return a, b

def write_problems_to_file(filename, problems):
    """Writes addition problems to a file in the required format."""
    with open(filename, 'w') as file:
        for problem in problems:
            file.write(f"{problem[0]} + {problem[1]}\n")

def write_answers_to_file(filename, answers):
    """Writes answers to a file."""
    with open(filename, 'w') as file:
        for answer in answers:
            file.write(f"{answer}\n")


input = 'data/inputs_addition_5digit.txt'
output = 'data/outputs_addition_5digit.txt'
def main(num_problems, low=0, high=99999):
    problems = [generate_addition_problem(low, high) for _ in range(num_problems)]
    write_problems_to_file(input, problems)
    answers = [sum(problem) for problem in problems]
    write_answers_to_file(output, answers)
    print(f'Addition problems and their answers have been written to {input} and {output}.')

if __name__ == "__main__":    
    main(500)
