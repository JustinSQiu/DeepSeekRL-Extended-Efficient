"""
Module for loading LLMs and their tokenizers from huggingface. 

"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase


def get_llm_tokenizer(model_name: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure a language model and its tokenizer.

    Args:
        model_name: Name or path of the pretrained model to load
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        tuple containing:
            - The loaded language model
            - The configured tokenizer for that model
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto", 
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False

    # for name, param in model.named_parameters():
    #     print(name)

    # for name, module in model.named_modules():
    #     print(name)

    
    return model, tokenizer

if __name__ == "__main__":
    pass
    # model, tokenizer = get_llm_tokenizer('Qwen/Qwen2.5-1.5B-Instruct', 'cuda')