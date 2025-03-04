"""
Hold all data sets 

"""

import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Any, List
import json



class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This class defines the interface that all dataset loaders should implement.
    Specific dataset loaders should inherit from this class and implement the
    required methods.
    
    Attributes:
        random (bool): If True, returns items randomly; if False, returns sequentially
        current_index (int): Current position for sequential access
    """
    
    def __init__(self, random: bool = False) -> None:
        self.random = random
        self.current_index = 0
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        pass
        
    @abstractmethod
    def __iter__(self) -> 'DataLoader':
        """Return self as iterator."""
        return self
        
    @abstractmethod
    def __next__(self) -> Any:
        """Return the next item(s) in the dataset."""
        pass


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()



SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""



class GSM8KLoader(DataLoader):
    """
    A loader class that provides iteration over GSM8K math problems.
    
    This class implements both sequential and random access to math problems through
    standard Python iterator protocols. It can be used to iterate over problems either
    in order or randomly, making it suitable for both training and evaluation.
    
    Attributes:
        questions (List[str]): List of math question strings
        answers (List[str]): List of corresponding answer strings
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
    """
    
    def __init__(self, questions: list[str], answers: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.questions = questions
        self.answers = answers
        self.pre_prompt = """You will be given a question that involves reasoning. You should reason carefully about the question, then provide your answer.
            It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:
            
            <reasoning>
            Your step-by-step reasoning process here
            </reasoning>
            <answer>
            Your final answer here
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. It is is extremely important that you do not put any text outside of these tags.

            Question: """
        self.system_prompt = SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.questions)
        
    def __iter__(self) -> 'GSM8KLoader':
        return self
        
    def __next__(self) -> tuple[str, str]:
        if self.current_index >= len(self.questions):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.questions) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.questions[idx], self.answers[idx]

    def reset(self):
        self.current_index = 0 


def build_gsm8k_dataloaders() -> Tuple[GSM8KLoader, GSM8KLoader]: 
    data = load_dataset('openai/gsm8k', 'main')["train"]

    questions = []
    parsed_answers = [] 
    for i in tqdm(range(len(data)), desc="Processing"):
        # Try to get answer - if is None dont use this sample 
        ans = extract_hash_answer(data[i]['answer'])
        if ans is None: 
            continue 
        else:
            questions.append(data[i]['question'])
            parsed_answers.append(ans)

    # Randomly split into train/test sets
    total_samples = len(questions)
    test_size = int(total_samples * 0.01)  # 10% for test set
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    # Convert to numpy arrays for easier indexing
    questions = np.array(questions)
    parsed_answers = np.array(parsed_answers)
    
    # Create boolean mask for test indices
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True
    
    # Split using boolean indexing
    test_questions = questions[test_mask]
    test_answers = parsed_answers[test_mask]
    train_questions = questions[~test_mask]
    train_answers = parsed_answers[~test_mask]

    # Setup data loaders 
    trainloader = GSM8KLoader(train_questions.tolist(), train_answers.tolist())
    testloader = GSM8KLoader(test_questions.tolist(), test_answers.tolist())
    
    return trainloader, testloader


class MatrixRREFLoader(DataLoader):
    """ A loader class for Gaussian elimination problems. Each example consists of a matrix (the question) and its RREF (the answer). """
    
    def __init__(self, matrices: List[str], RREFs: List[str], random: bool = False) -> None:
        super().__init__()
        self.random = random
        self.matrices = matrices
        self.RREFs = RREFs
        self.pre_prompt = """You will be given a question that involves computing the reduced row-echelon form of a matrix. You should reason carefully about the question, then provide your answer.
            It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:


            <reasoning>
            Your step-by-step reasoning process here
            </reasoning>
            <answer>
            Your final answer here
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. 
            It is is extremely important you answer in this way - do not put any information or text outside of these tags!
            Your <answer> tag should only contain the reduced row-echelon form of the matrix in this form:
            [[a, b], [c, d]]

            Question: """

        self.system_prompt = SYSTEM_PROMPT
        self.current_index = 0

    def __len__(self) -> int:
        return len(self.matrices)

    def __iter__(self) -> 'MatrixRREFLoader':
        return self

    def __next__(self) -> Tuple[str, str]:
        if self.current_index >= len(self.matrices):
            raise StopIteration
        idx = random.randint(0, len(self.matrices) - 1) if self.random else self.current_index
        if not self.random:
            self.current_index += 1
        return self.matrices[idx], self.RREFs[idx]

    def reset(self):
        self.current_index = 0


def build_matrix_RREF_dataloaders() -> Tuple[DataLoader, DataLoader]:
    # Replace the following with code to load your matrix RREF dataset. 
    # For illustration, we assume matrices and RREFs are stored as lists of strings.

    with open('data/inputs.txt', 'r') as file:
        lines = file.readlines()
    matrices = [line.strip() for line in lines]

    with open('data/outputs.txt', 'r') as file:
        lines = file.readlines()
    RREFs = [line.strip() for line in lines]

    # Use a simple split (or any strategy you prefer) for train/test
    total = len(matrices)
    test_size = int(total * 0.1)  # e.g., 10% test set
    indices = list(range(total))
    random.shuffle(indices)
    test_idx = set(indices[:test_size])
    
    train_matrices = [m for i, m in enumerate(matrices) if i not in test_idx]
    train_RREFs = [inv for i, inv in enumerate(RREFs) if i not in test_idx]
    test_matrices = [m for i, m in enumerate(matrices) if i in test_idx]
    test_RREFs = [inv for i, inv in enumerate(RREFs) if i in test_idx]
    
    return MatrixRREFLoader(train_matrices, train_RREFs), MatrixRREFLoader(test_matrices, test_RREFs)

class AdditionLoader(DataLoader):
    """
    A loader class that provides iteration over addition problems.
    
    This class implements both sequential and random access to addition problems through
    standard Python iterator protocols. It can be used to iterate over problems either
    in order or randomly, making it suitable for both training and evaluation.
    
    Attributes:
        problems (List[Tuple[int, int]]): List of addition problem tuples (a, b)
        answers (List[int]): List of corresponding answer strings (a + b)
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
    """
    
    def __init__(self, problems: List[Tuple[int, int]], answers: List[int], random: bool = False) -> None:
        super().__init__(random)
        self.problems = problems
        self.answers = answers
        self.pre_prompt = """You will be given a math addition problem. You should reason carefully about the question, then provide your answer.
            It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:
            
            <reasoning>
            Your step-by-step reasoning process here
            </reasoning>
            <answer>
            Your final answer here
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. It is is extremely important that you do not put any text outside of these tags.

            Question: """
        self.system_prompt = SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.problems)
        
    def __iter__(self) -> 'AdditionLoader':
        return self
        
    def __next__(self) -> Tuple[str, str]:
        if self.current_index >= len(self.problems):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.problems) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        a, b = self.problems[idx]
        answer = self.answers[idx]
        question = f"{a} + {b}"
        
        return question, str(answer)

    def reset(self):
        self.current_index = 0


def build_addition_dataloaders() -> Tuple['AdditionLoader', 'AdditionLoader']:
    # Read and parse the addition problems
    with open('data/inputs_addition_5digit.txt', 'r') as file:
        lines = file.readlines()

    # Convert each line into a tuple of integers (e.g. "24525 + 57305" -> (24525, 57305))
    problems = []
    for line in lines:
        parts = line.strip().split('+')
        if len(parts) != 2:
            continue
        a = int(parts[0].strip())
        b = int(parts[1].strip())
        problems.append((a, b))

    # Read and convert outputs to integers
    with open('data/outputs_addition_5digit.txt', 'r') as file:
        outputs = [int(line.strip()) for line in file.readlines()]

    total = len(problems)
    test_size = int(total * 0.1)  # 10% test set
    indices = list(range(total))
    random.shuffle(indices)
    test_idx = set(indices[:test_size])

    # Split data into training and test sets
    train_problems = [problems[i] for i in range(total) if i not in test_idx]
    train_outputs = [outputs[i] for i in range(total) if i not in test_idx]
    test_problems = [problems[i] for i in range(total) if i in test_idx]
    test_outputs = [outputs[i] for i in range(total) if i in test_idx]

    # Return AdditionLoader instances directly
    train_dataset = AdditionLoader(train_problems, train_outputs)
    test_dataset = AdditionLoader(test_problems, test_outputs)

    return train_dataset, test_dataset


class MatrixInverseLoader(DataLoader):
    """ Loader for matrix inversion problems.
    
    Each example consists of a 5x5 matrix (as the question) and its inverse (as the answer).
    The question is formatted with a pre_prompt instructing the model to include all output within
    <think> and <answer> tags.
    """
    
    def __init__(self, matrices: List[str], inverses: List[str], random: bool = False) -> None:
        super().__init__(random)
        self.random = random
        self.matrices = matrices
        self.inverses = inverses
        self.pre_prompt = """You will be asked to compute the inverse of a matrix. You should reason carefully about the question, then provide your answer.
            It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:
            
            <reasoning>
            Your step-by-step reasoning process here
            </reasoning>
            <answer>
            Your final answer here, in the form of a Python list of lists.
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. It is is extremely important that you do not put any text outside of these tags.

            Question: """

        self.system_prompt = SYSTEM_PROMPT
        self.current_index = 0

    def __len__(self) -> int:
        return len(self.matrices)

    def __iter__(self) -> 'MatrixInverseLoader':
        return self

    def __next__(self) -> Tuple[str, str]:
        if self.current_index >= len(self.matrices):
            raise StopIteration
        idx = random.randint(0, len(self.matrices) - 1) if self.random else self.current_index
        if not self.random:
            self.current_index += 1
        question = self.matrices[idx]
        answer = self.inverses[idx]
        return question, answer

    def reset(self):
        self.current_index = 0


def build_matrix_inverse_dataloaders(total_samples: int = 1000, test_ratio: float = 0.1) -> Tuple[MatrixInverseLoader, MatrixInverseLoader]:
    train_matrices = []
    train_inverses = []
    test_matrices = []
    test_inverses = []

    samples = []
    while len(samples) < total_samples:
        # Generate a random 5x5 matrix with entries in [-10, 10]
        mat = np.random.randint(-10, 10, (5, 5))
        # Check for invertibility (determinant not too close to zero)
        if abs(np.linalg.det(mat)) < 1e-3:
            continue
        # Format matrix as a Python list of lists string
        mat_list = mat.tolist()
        mat_str = str(mat_list)
        # Compute inverse and format as string (you might want to round values)
        inv_mat = np.linalg.inv(mat)
        inv_list = inv_mat.tolist()
        inv_str = str(inv_list)
        samples.append((mat_str, inv_str))

    # Split samples into train and test sets
    random.shuffle(samples)
    test_size = int(total_samples * test_ratio)
    train_samples = samples[test_size:]
    test_samples = samples[:test_size]

    train_matrices, train_inverses = zip(*train_samples)
    test_matrices, test_inverses = zip(*test_samples)

    return MatrixInverseLoader(list(train_matrices), list(train_inverses)), MatrixInverseLoader(list(test_matrices), list(test_inverses))


def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('gsm8k', 'addition', etc.)
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() == 'gsm8k':
        return build_gsm8k_dataloaders()
    elif dataset_name.lower() == 'matrix_rref':
        return build_matrix_RREF_dataloaders()
    elif dataset_name.lower() == 'addition':
        return build_addition_dataloaders()
    elif dataset_name.lower() == 'matrix_inverse':
        return build_matrix_inverse_dataloaders()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


if __name__ == "__main__": 
    trainloader, testloader = get_dataloaders('gsm8k')
