"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    
    The main methods that need to be implemented are:
    - compute_rewards: Computes rewards for a batch of completions
    - get_reward_breakdown: Converts raw reward scores to a labeled dictionary
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            prompts: List of prompt messages in chat format
                    [{"role": "user", "content": "..."}, ...]
            completions: List of completion messages in chat format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Ground truth answer(s) for the prompts
            device: Device to place tensors on ("cpu" or "cuda")
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
                            containing individual reward function scores
            metrics: Dictionary of aggregated metrics including mean rewards
                    per function and total reward
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward scores tensor to a labeled dictionary.
        
        Args:
            reward_scores: Tensor of raw scores from compute_rewards
            
        Returns:
            Dictionary mapping reward function names to their scores
        """
        pass


class GSM8kEvaluator(RewardEvaluator):
    """
    Reward evaluator for the GSM8K math problem dataset.
    
    Implements reward functions for:
    - Answer correctness
    - Integer format validation
    - XML formatting (strict and soft)
    - XML tag counting
    """
    
    def __init__(self):
        self.num_reward_functions = 5
    
    def _extract_xml_answer(self, text: str) -> str:
        """Extract answer from XML tags."""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    def _correctness_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for correct answer."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

    def _int_format_reward(self, completions) -> List[float]:
        """Reward for integer format."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted]

    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.match(pattern, r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.match(pattern, r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if text.count("<reasoning>\n") == 1: count += 0.125
            if text.count("\n</reasoning>\n") == 1: count += 0.125
            if text.count("\n<answer>\n") == 1:
                count += 0.125
                count -= len(text.split("\n</answer>\n")[-1])*0.001
            if text.count("\n</answer>") == 1:
                count += 0.125
                count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
            return count
            
        responses = [completion[0]["content"] for completion in completions]
        return [count_xml(r) for r in responses]

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute all reward functions
        all_scores = [
            self._correctness_reward(prompts, completions, answer),
            self._int_format_reward(completions),
            self._strict_format_reward(completions),
            self._soft_format_reward(completions),
            self._xml_count_reward(completions)
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate accuracy (perfect correctness score)
        correctness_scores = rewards_per_func[:, 0]  # First reward function is correctness
        num_perfect = (correctness_scores == 2.0).sum().item()
        accuracy = num_perfect / num_completions
        
        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/int_reward_func": reward_per_func[1].item(), 
            "rewards/strict_format_reward_func": reward_per_func[2].item(),
            "rewards/soft_format_reward_func": reward_per_func[3].item(),
            "rewards/xmlcount_reward_func": reward_per_func[4].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'correctness': reward_scores[0].item(),
            'integer_format': reward_scores[1].item(),
            'strict_format': reward_scores[2].item(),
            'soft_format': reward_scores[3].item(),
            'xml_count': reward_scores[4].item()
        }


class MatrixRREFEvaluator(RewardEvaluator):
    """ Evaluator for matrix Gaussian elimination tasks.

    The evaluator parses the generated answer, which should be formatted as:

      <think>
      ...internal computation steps...
      </think>
      <answer>
      ...the computed RREF (as a Python list of lists)...
      </answer>

    It then compares the predicted RREF with the ground truth. In addition,
    it verifies that the output adheres to the expected format.
    """

    def __init__(self, tol: float = 1e-3):
        self.tol = tol  # tolerance for numerical error
        self.num_reward_functions = 5

    def _extract_answer(self, text: str) -> str:
        """Extract answer from <answer> tags."""
        try:
            answer = text.split("<answer>")[-1].split("</answer>")[0]
            return answer.strip()
        except Exception:
            return ""
    
    def _extract_think(self, text: str) -> str:
        """Extract think block from <think> tags."""
        try:
            think = text.split("<think>")[-1].split("</think>")[0]
            return think.strip()
        except Exception:
            return ""

    def _correctness_reward(self, completions, answer) -> List[float]:
        """Reward for correct matrix Gaussian elimination."""
        
        rewards = []
        
        for i, completion in enumerate(completions):
            text = completion[0]['content']
            pred_str = self._extract_answer(text)
            
            try:
                # Evaluate the predicted matrix from the answer string.
                pred_matrix = eval(pred_str)
                
                # Determine the corresponding ground truth; if answer is a list, use the i-th element.
                true_matrix = eval(answer[i]) if isinstance(answer, list) else eval(answer)
                
                pred_tensor = torch.tensor(pred_matrix, dtype=torch.float32)
                true_tensor = torch.tensor(true_matrix, dtype=torch.float32)
                
                # Check that shapes match before comparing values.
                if pred_tensor.shape != true_tensor.shape:
                    reward = 0.0
                elif torch.allclose(pred_tensor, true_tensor, atol=self.tol):
                    reward = 2.0
                else:
                    reward = 0.0
            except Exception:
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards

    def _valid_format_reward(self, completions) -> List[float]:
        """
        Reward for ensuring the predicted RREF is in a valid format
        (i.e., a Python list-of-lists with numeric entries).
        """
        rewards = []
        for completion in completions:
            text = completion[0]['content']
            ans_str = self._extract_answer(text)
            try:
                matrix = eval(ans_str)
                if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
                    valid = all(all(isinstance(x, (int, float)) for x in row) for row in matrix)
                    rewards.append(0.5 if valid else 0.0)
                else:
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    def _strict_format_reward(self, completions) -> List[float]:
        """
        Reward for strict adherence to the expected format:
        <think>
        ...content...
        </think>
        <answer>
        ...content...
        </answer>
        (with proper newlines).
        """
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
        rewards = []
        for completion in completions:
            text = completion[0]['content']
            match = bool(re.match(pattern, text, re.DOTALL))
            rewards.append(0.5 if match else 0.0)
        return rewards

    def _soft_format_reward(self, completions) -> List[float]:
        """
        Reward for a looser check on formatting: the output should contain
        both <think>...</think> and <answer>...</answer> blocks.
        """
        pattern = r"<think>.*?</think>.*<answer>.*?</answer>"
        rewards = []
        for completion in completions:
            text = completion[0]['content']
            match = bool(re.search(pattern, text, re.DOTALL))
            rewards.append(0.5 if match else 0.0)
        return rewards

    def _tag_count_reward(self, completions) -> List[float]:
        """
        Reward based on the correct count of tags. The output should contain
        exactly one <think>, one </think>, one <answer>, and one </answer> tag.
        """
        rewards = []
        for completion in completions:
            text = completion[0]['content']
            count_think_open = text.count("<think>")
            count_think_close = text.count("</think>")
            count_answer_open = text.count("<answer>")
            count_answer_close = text.count("</answer>")
            if count_think_open == 1 and count_think_close == 1 and count_answer_open == 1 and count_answer_close == 1:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Compute rewards using our defined functions
        correctness = self._correctness_reward(completions, answer)
        valid_format = self._valid_format_reward(completions)
        strict_format = self._strict_format_reward(completions)
        soft_format = self._soft_format_reward(completions)
        tag_count = self._tag_count_reward(completions)
        
        all_scores = [correctness, valid_format, strict_format, soft_format, tag_count]
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute aggregated metrics
        reward_per_func_avg = rewards_per_func.mean(0)
        num_perfect = sum(1 for r in correctness if r == 2.0)
        accuracy = num_perfect / num_completions if num_completions > 0 else 0.0
        
        metrics = {
            "rewards/correctness_reward_func": reward_per_func_avg[0].item(),
            "rewards/int_reward_func": reward_per_func_avg[1].item(),
            "rewards/strict_format_reward_func": reward_per_func_avg[2].item(),
            "rewards/soft_format_reward_func": reward_per_func_avg[3].item(),
            "rewards/xmlcount_reward_func": reward_per_func_avg[4].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        return {
            "correctness": reward_scores[0].item(),
            "integer_format": reward_scores[1].item(),
            "strict_format": reward_scores[2].item(),
            "soft_format": reward_scores[3].item(),
            "xml_count": reward_scores[4].item()
        }
    
class EnhancedMatrixRREFEvaluator(RewardEvaluator):
    """
    An enhanced evaluator for matrix RREF tasks that rewards:
      - Binary correctness (full matrix match)
      - Row-by-row correctness (partial credit for each correct row)
      - Continuous reward based on the Frobenius norm (overall numerical closeness)
      - Format adherence via tag counting
      - Length of the reasoning process in the <think></think> block
    """
    def __init__(self, tol: float = 1e-3, frobenius_scale: float = 1.0):
        self.tol = tol
        self.frobenius_scale = frobenius_scale
        # We'll use 8 reward functions:
        # 0: Binary correctness, 1: Row-by-row reward,
        # 2: Frobenius norm reward, 3: Valid format reward,
        # 4: Strict format reward, 5: Soft format reward,
        # 6: Tag count reward, 7: <think> length reward.
        self.num_reward_functions = 8

    def _extract_answer(self, text: str) -> str:
        """Extract answer from <answer> tags."""
        try:
            return text.split("<answer>")[-1].split("</answer>")[0].strip()
        except Exception:
            return ""

    def _extract_think(self, text: str) -> str:
        """Extract the content within <think> tags."""
        try:
            return text.split("<think>")[-1].split("</think>")[0].strip()
        except Exception:
            return ""

    def _binary_correctness_reward(self, completions, answer) -> List[float]:
        """Reward 2.0 if the entire matrix matches the true RREF (within tolerance), else 0."""
        rewards = []
        for i, completion in enumerate(completions):
            text = completion[0]['content']
            ans_str = self._extract_answer(text)
            try:
                pred_matrix = torch.tensor(eval(ans_str), dtype=torch.float32)
                true_matrix = torch.tensor(eval(answer[i]) if isinstance(answer, list) else eval(answer), dtype=torch.float32)
                if pred_matrix.shape != true_matrix.shape:
                    reward = 0.0
                elif torch.allclose(pred_matrix, true_matrix, atol=self.tol):
                    reward = 2.0
                else:
                    reward = 0.0
            except Exception:
                reward = 0.0
            rewards.append(reward)
        return rewards

    def _row_by_row_reward(self, completions, answer) -> List[float]:
        """Compute a reward based on the fraction of rows that match the true RREF."""
        rewards = []
        for i, completion in enumerate(completions):
            text = completion[0]['content']
            ans_str = self._extract_answer(text)
            try:
                pred_matrix = eval(ans_str)
                true_matrix = eval(answer[i]) if isinstance(answer, list) else eval(answer)
                # Ensure matrices are lists of lists
                if not (isinstance(pred_matrix, list) and isinstance(true_matrix, list)):
                    rewards.append(0.0)
                    continue
                num_rows = min(len(pred_matrix), len(true_matrix))
                row_rewards = 0.0
                for r in range(num_rows):
                    pred_row = torch.tensor(pred_matrix[r], dtype=torch.float32)
                    true_row = torch.tensor(true_matrix[r], dtype=torch.float32)
                    if pred_row.shape != true_row.shape:
                        continue
                    if torch.allclose(pred_row, true_row, atol=self.tol):
                        row_rewards += 1.0
                # Scale the reward to a maximum of 2.0
                rewards.append(2.0 * (row_rewards / num_rows) if num_rows > 0 else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    def _frobenius_reward(self, completions, answer) -> List[float]:
        """
        Compute a reward based on the Frobenius norm of the difference.
        The reward is higher when the error is lower.
        """
        rewards = []
        for i, completion in enumerate(completions):
            text = completion[0]['content']
            ans_str = self._extract_answer(text)
            try:
                pred_matrix = torch.tensor(eval(ans_str), dtype=torch.float32)
                true_matrix = torch.tensor(eval(answer[i]) if isinstance(answer, list) else eval(answer), dtype=torch.float32)
                if pred_matrix.shape != true_matrix.shape:
                    error = float('inf')
                else:
                    error = torch.norm(pred_matrix - true_matrix, p='fro').item()
                # Map the error to a reward (example: reward decreases linearly with error)
                reward = max(0.0, 2.0 - self.frobenius_scale * error)
            except Exception:
                reward = 0.0
            rewards.append(reward)
        return rewards

    def _valid_format_reward(self, completions) -> List[float]:
        """
        Reward for ensuring the predicted RREF is in a valid Python list-of-lists format.
        """
        rewards = []
        for completion in completions:
            text = completion[0]['content']
            ans_str = self._extract_answer(text)
            try:
                matrix = eval(ans_str)
                if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
                    valid = all(all(isinstance(x, (int, float)) for x in row) for row in matrix)
                    rewards.append(0.5 if valid else 0.0)
                else:
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    def _strict_format_reward(self, completions) -> List[float]:
        """
        Reward for strict adherence to the expected format:
        <think>
        ...content...
        </think>
        <answer>
        ...content...
        </answer>
        with proper newlines.
        """
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
        rewards = []
        for completion in completions:
            text = completion[0]['content']
            match = bool(re.match(pattern, text, re.DOTALL))
            rewards.append(0.5 if match else 0.0)
        return rewards

    def _soft_format_reward(self, completions) -> List[float]:
        """
        Reward for a looser check on formatting: the output should contain both <think>...</think>
        and <answer>...</answer> blocks.
        """
        pattern = r"<think>.*?</think>.*<answer>.*?</answer>"
        rewards = []
        for completion in completions:
            text = completion[0]['content']
            match = bool(re.search(pattern, text, re.DOTALL))
            rewards.append(0.5 if match else 0.0)
        return rewards

    def _tag_count_reward(self, completions) -> List[float]:
        """
        Reward based on the correct count of tags. The output should contain exactly one
        <think>, one </think>, one <answer>, and one </answer> tag.
        """
        rewards = []
        for completion in completions:
            text = completion[0]['content']
            count_think_open = text.count("<think>")
            count_think_close = text.count("</think>")
            count_answer_open = text.count("<answer>")
            count_answer_close = text.count("</answer>")
            if count_think_open == 1 and count_think_close == 1 and count_answer_open == 1 and count_answer_close == 1:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards

    def _think_length_reward(self, completions) -> List[float]:
        """
        Reward based on the length of the content within the <think></think> block.
        The reward scales linearly with the number of characters up to a maximum of 0.5.
        """
        rewards = []
        target_length = 100  # Target number of characters for full reward.
        for completion in completions:
            text = completion[0]['content']
            think_text = self._extract_think(text)
            # Scale reward: if think_text is shorter than target_length, get proportionally less reward.
            reward = min(0.5, (len(think_text) / target_length) * 0.5)
            rewards.append(reward)
        return rewards

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute each reward function
        binary_rewards = self._binary_correctness_reward(completions, answer)
        row_rewards = self._row_by_row_reward(completions, answer)
        frob_rewards = self._frobenius_reward(completions, answer)
        valid_format_rewards = self._valid_format_reward(completions)
        strict_format_rewards = self._strict_format_reward(completions)
        soft_format_rewards = self._soft_format_reward(completions)
        tag_count_rewards = self._tag_count_reward(completions)
        think_length_rewards = self._think_length_reward(completions)

        all_scores = [
            binary_rewards,
            row_rewards,
            frob_rewards,
            valid_format_rewards,
            strict_format_rewards,
            soft_format_rewards,
            tag_count_rewards,
            think_length_rewards
        ]
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)

        # Compute aggregated metrics
        reward_per_func_avg = rewards_per_func.mean(0)
        num_perfect = sum(1 for r in binary_rewards if r == 2.0)
        accuracy = num_perfect / num_completions if num_completions > 0 else 0.0

        metrics = {
            "rewards/binary_correctness": reward_per_func_avg[0].item(),
            "rewards/row_by_row": reward_per_func_avg[1].item(),
            "rewards/frobenius": reward_per_func_avg[2].item(),
            "rewards/valid_format": reward_per_func_avg[3].item(),
            "rewards/strict_format": reward_per_func_avg[4].item(),
            "rewards/soft_format": reward_per_func_avg[5].item(),
            "rewards/tag_count": reward_per_func_avg[6].item(),
            "rewards/think_length": reward_per_func_avg[7].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        return {
            "binary_correctness": reward_scores[0].item(),
            "row_by_row": reward_scores[1].item(),
            "frobenius": reward_scores[2].item(),
            "valid_format": reward_scores[3].item(),
            "strict_format": reward_scores[4].item(),
            "soft_format": reward_scores[5].item(),
            "tag_count": reward_scores[6].item(),
            "think_length": reward_scores[7].item()
        }

# Example helper function to get the evaluator
def get_evaluator(name: str) -> RewardEvaluator:
    """ Get the appropriate reward evaluator for a given task.

    Args:
        name: Name of the task/dataset to get evaluator for
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "gsm8k":
        return GSM8kEvaluator()
    elif name.lower() == "matrix_rref":
        return MatrixRREFEvaluator()
    elif name.lower() == "enhanced_matrix_rref":
        return EnhancedMatrixRREFEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")
