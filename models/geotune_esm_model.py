"""
ESM2 Model with LoRA integration for constraint learning
"""
import torch
import torch.nn as nn
from transformers import EsmModel, EsmConfig, EsmTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging

logger = logging.getLogger(__name__)


class GeoTuneESMModel(nn.Module):
    """
    ESM2 model with integrated geometric constraint capabilities
    """
    def __init__(
        self,
        base_model,
        config,
        lora_config=None,
        constraint_weight=0.1
    ):
        super().__init__()

        self.config = config

        # If LoRA is not already applied, apply it now
        if lora_config is not None and not isinstance(base_model, PeftModel):
            self.model = get_peft_model(base_model, lora_config)
            logger.info(f"Applied LoRA with config: {lora_config}")
        else:
            self.model = base_model
        
        # Add language modeling head for MLM task
        hidden_size = self.config.hidden_size
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size, bias=False)

        # Initialize LM head weights from word embeddings
        # Access base model through self.model (or self.model.base_model for PEFT)
        actual_base = self.model.base_model if hasattr(self.model, 'base_model') else self.model

        try:
            # Access the embeddings properly for ESM2 model
            if hasattr(actual_base, 'embeddings') and hasattr(actual_base.embeddings, 'word_embeddings'):
                self.lm_head.weight = actual_base.embeddings.word_embeddings.weight
            elif hasattr(actual_base, 'encoder') and hasattr(actual_base.encoder, 'embed_tokens'):
                # For some transformer variants
                self.lm_head.weight = actual_base.encoder.embed_tokens.weight
            elif hasattr(actual_base, 'shared'):
                # For models with shared embeddings
                self.lm_head.weight = actual_base.shared.weight
            else:
                logger.warning("Could not find appropriate embeddings to tie with LM head, using random initialization.")
        except AttributeError as e:
            logger.warning(f"Could not tie LM head weights: {e}. Using random initialization.")

        # Add constraint-specific layers
        self.constraint_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.constraint_activation = nn.Tanh()

        # Constraint weight for loss combination
        self.constraint_weight = constraint_weight

        # Freeze all base model parameters when LoRA is applied
        # Only LoRA parameters and lm_head should be trainable
        if lora_config is not None:
            for param in self.model.parameters():
                param.requires_grad = False

            # Enable gradients only for LoRA parameters
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                    logger.info(f"Enabled gradients for LoRA parameter: {name}")

        # The LM head should be trainable for the MLM task
        for param in self.lm_head.parameters():
            param.requires_grad = True
            logger.info("LM head parameters set to trainable")

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
        geometric_features=None,
    ):
        outputs = self.model(
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
        
        sequence_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        
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
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs['sequence_output']

    def save_lora_adapters(self, save_path):
        """Saves the LoRA adapter weights."""
        if isinstance(self.model, PeftModel):
            try:
                self.model.save_pretrained(save_path)
                logger.info(f"LoRA adapters saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save LoRA adapters: {e}")
                raise
        else:
            logger.warning("Model is not a PeftModel, so no LoRA adapters to save.")


def create_lora_config(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=None,
    bias="none"
):
    if target_modules is None:
        target_modules = [
            "query", "key", "value", "dense",
            "intermediate.dense", "output.dense"
        ]
    
    # For ESM models, use TOKEN_CLS which is more appropriate for encoder models
    config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )
    
    return config


def load_esm_with_lora(model_name="facebook/esm2_t30_150M_UR50D", lora_params=None, lora_weights_path=None):
    """
    Load ESM model with optional LoRA configuration and weights.

    Args:
        model_name (str): The name of the pre-trained ESM model.
        lora_params (dict, optional): Parameters for LoRA configuration.
        lora_weights_path (str, optional): Path to pre-trained LoRA adapter weights.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    if lora_params is None:
        lora_params = {}

    config = EsmConfig.from_pretrained(model_name)
    base_model = EsmModel.from_pretrained(model_name, config=config, torch_dtype=torch.float32)

    if lora_weights_path:
        logger.info(f"Loading LoRA adapters from {lora_weights_path}")
        # Load the PEFT model directly from the path
        peft_model = PeftModel.from_pretrained(base_model, lora_weights_path)
        model = GeoTuneESMModel(base_model=peft_model, config=config)
    else:
        logger.info("Initializing new LoRA adapters")
        lora_config = create_lora_config(**lora_params)
        model = GeoTuneESMModel(base_model=base_model, config=config, lora_config=lora_config)

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    
    return model, tokenizer