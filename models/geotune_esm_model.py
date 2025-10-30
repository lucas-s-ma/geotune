"""
ESM2 Model with LoRA integration for constraint learning
"""
import torch
import torch.nn as nn
from transformers import EsmModel, EsmConfig, EsmTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)


class ESMWithConstraints(nn.Module):
    """
    ESM2 model with integrated geometric constraint capabilities
    """
    def __init__(
        self, 
        model_name="facebook/esm2_t30_150M_UR50D", 
        lora_config=None,
        constraint_weight=0.1
    ):
        super().__init__()
        
        # Load base ESM model
        self.config = EsmConfig.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Apply LoRA if configuration is provided
        if lora_config is not None:
            self.esm = get_peft_model(self.esm, lora_config)
            logger.info(f"Applied LoRA with config: {lora_config}")
        
        # Add constraint-specific layers
        hidden_size = self.config.hidden_size
        self.constraint_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.constraint_activation = nn.Tanh()
        
        # Constraint weight for loss combination
        self.constraint_weight = constraint_weight
        
        # Add language modeling head for MLM task
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size, bias=False)
        
        # Initialize LM head weights from word embeddings
        self.lm_head.weight = self.esm.embeddings.word_embeddings.weight
        
        # Freeze all base model parameters when LoRA is applied
        # Only LoRA parameters and lm_head should be trainable
        if lora_config is not None:
            for param in self.esm.parameters():
                param.requires_grad = False
            
            # Enable gradients only for LoRA parameters
            for name, param in self.esm.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                    logger.info(f"Enabled gradients for LoRA parameter: {name}")
        
        # The LM head should be trainable for the MLM task
        for param in self.lm_head.parameters():
            param.requires_grad = True
            logger.info("LM head parameters set to trainable")
        
        # Add constraint-specific layers
        hidden_size = self.config.hidden_size
        self.constraint_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.constraint_activation = nn.Tanh()
        
        # Constraint weight for loss combination
        self.constraint_weight = constraint_weight
        
        # The get_peft_model function properly handles freezing base model parameters
        # when LoRA is applied, only keeping LoRA adapter parameters trainable
        
        # Add language modeling head for MLM task
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size, bias=False)
        
        # Initialize LM head weights from word embeddings
        self.lm_head.weight = self.esm.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        geometric_features=None,  # Additional geometric features (this can be the CA coordinates passed but not used in this implementation)
    ):
        # Forward pass through ESM
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Extract last hidden states
        sequence_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        
        # Apply constraint projection
        constraint_features = self.constraint_activation(
            self.constraint_proj(sequence_output)
        )
        
        return {
            'sequence_output': sequence_output,
            'constraint_features': constraint_features,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }

    def get_sequence_embeddings(self, input_ids, attention_mask=None):
        """
        Get sequence embeddings from the model
        """
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs['sequence_output']

    def save_lora_adapters(self, save_path):
        """
        Save only LoRA adapter weights
        """
        if hasattr(self.esm, 'save_pretrained'):
            self.esm.save_pretrained(save_path)
        else:
            raise AttributeError("Model doesn't support saving LoRA adapters")

    def load_lora_adapters(self, load_path):
        """
        Load LoRA adapter weights
        """
        if hasattr(self.esm, 'from_pretrained'):
            # Reload the PEFT model from saved adapters
            self.esm = self.__class__.from_pretrained(load_path)
        else:
            raise AttributeError("Model doesn't support loading LoRA adapters")


def create_lora_config(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=None,
    bias="none"
):
    """
    Create LoRA configuration for ESM models
    
    Args:
        r: LoRA attention dimension
        lora_alpha: Scaling factor
        lora_dropout: Dropout rate for LoRA layers
        target_modules: List of modules to apply LoRA to
        bias: Bias setting for LoRA
    """
    if target_modules is None:
        # Default target modules for ESM2 attention and feed-forward layers
        target_modules = [
            "query",
            "key", 
            "value",
            "dense",  # output projection in attention
            "intermediate.dense",  # intermediate layer in feed-forward
            "output.dense"  # output layer in feed-forward
        ]
    
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # For ESM without specific head
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )
    
    return config


def load_esm_with_lora(model_name="facebook/esm2_t30_150M_UR50D", lora_params=None):
    """
    Load ESM model with optional LoRA configuration
    """
    if lora_params is None:
        lora_params = {}
        
    lora_config = create_lora_config(**lora_params)
    model = ESMWithConstraints(
        model_name=model_name,
        lora_config=lora_config
    )
    
    # Get tokenizer
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    
    return model, tokenizer