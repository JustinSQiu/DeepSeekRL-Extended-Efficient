"""
Hold all data sets 

"""

import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Any, List



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

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. 
            It is is extremely important you answer in this way - do not put any information or text outside of these tags!

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


def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('gsm8k' currently supported)
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() == 'gsm8k':
        return build_gsm8k_dataloaders()
    elif dataset_name.lower() == 'matrix_inversion':
        return build_matrix_inversion_dataloaders()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


class MatrixInversionLoader(DataLoader):
    """ A loader class for matrix inversion problems. Each example consists of a matrix (the question) and its inverse (the answer). """
    
    def __init__(self, matrices: List[str], inverses: List[str], random: bool = False) -> None:
        super().__init__()
        self.random = random
        self.matrices = matrices
        self.inverses = inverses
        self.system_prompt = (
            "You will be given a matrix. Your task is to compute its inverse. "
            "Return your answer using the following format:\n"
            "<inverse>\n"
            "Your computed inverse here (as a 2D list or formatted matrix)\n"
            "</inverse>\n"
            "Do not include any text outside the tags."
        )
        self.current_index = 0

    def __len__(self) -> int:
        return len(self.matrices)

    def __iter__(self) -> 'MatrixInversionLoader':
        return self

    def __next__(self) -> Tuple[str, str]:
        if self.current_index >= len(self.matrices):
            raise StopIteration
        idx = random.randint(0, len(self.matrices) - 1) if self.random else self.current_index
        if not self.random:
            self.current_index += 1
        return self.matrices[idx], self.inverses[idx]

    def reset(self):
        self.current_index = 0


def build_matrix_inversion_dataloaders() -> Tuple[DataLoader, DataLoader]:
    # Replace the following with code to load your matrix inversion dataset. 
    # For illustration, we assume matrices and inverses are stored as lists of strings.
    
    matrices = [
        "[[1, 2], [3, 4]]", 
        "[[2, 0], [0, 2]]", 
        # ... add more examples
    ]
    inverses = [
        "[[-2.0, 1.0], [1.5, -0.5]]", 
        "[[0.5, 0], [0, 0.5]]", 
        # ... corresponding inverses
    ]
    
    # Use a simple split (or any strategy you prefer) for train/test
    total = len(matrices)
    test_size = int(total * 0.1)  # e.g., 10% test set
    indices = list(range(total))
    random.shuffle(indices)
    test_idx = set(indices[:test_size])
    
    train_matrices = [m for i, m in enumerate(matrices) if i not in test_idx]
    train_inverses = [inv for i, inv in enumerate(inverses) if i not in test_idx]
    test_matrices = [m for i, m in enumerate(matrices) if i in test_idx]
    test_inverses = [inv for i, inv in enumerate(inverses) if i in test_idx]
    
    return MatrixInversionLoader(train_matrices, train_inverses), MatrixInversionLoader(test_matrices, test_inverses)


if __name__ == "__main__": 
    trainloader, testloader = get_dataloaders('gsm8k')
