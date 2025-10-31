"""
ESM2 Model with LoRA integration for constraint learning
"""
import torch
import torch.nn as nn
from transformers import EsmModel, EsmConfig, EsmTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging

logger = logging.getLogger(__name__)


class ESMWithConstraints(nn.Module):
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
        self.model = base_model
        
        # Add language modeling head for MLM task
        hidden_size = self.config.hidden_size
        self.lm_head = nn.Linear(hidden_size, self.config.vocab_size, bias=False)
        
        # Initialize LM head weights from word embeddings BEFORE applying LoRA
        try:
            # Access the embeddings properly for ESM2 model
            if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
                self.lm_head.weight = self.model.embeddings.word_embeddings.weight
            elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'embed_tokens'):
                # For some transformer variants
                self.lm_head.weight = self.model.encoder.embed_tokens.weight
            elif hasattr(self.model, 'shared'):
                # For models with shared embeddings
                self.lm_head.weight = self.model.shared.weight
            else:
                # Try to find embeddings using different approach
                embedding_weights = None
                if hasattr(self.model, 'embeddings'):
                    if hasattr(self.model.embeddings, 'word_embeddings'):
                        embedding_weights = self.model.embeddings.word_embeddings.weight
                    elif hasattr(self.model.embeddings, 'tokens'):
                        embedding_weights = self.model.embeddings.tokens.weight
                    elif hasattr(self.model.embeddings, 'token_embeddings'):
                        embedding_weights = self.model.embeddings.token_embeddings.weight
                    elif hasattr(self.model.embeddings, 'decoder'):
                        embedding_weights = self.model.embeddings.decoder.weight
                    elif hasattr(self.model, 'word_embeddings'):
                        embedding_weights = self.model.word_embeddings.weight
                    elif hasattr(self.model, 'shared'):
                        embedding_weights = self.model.shared.weight
                    elif hasattr(self.model, 'wte'):  # GPT-style embedding
                        embedding_weights = self.model.wte.weight
                    else:
                        # Find any parameter that has the right shape for embeddings
                        vocab_size, hidden_size = self.lm_head.weight.shape
                        for name, param in self.model.named_parameters():
                            if param.shape == (vocab_size, hidden_size) and 'embed' in name:
                                embedding_weights = param
                                break
                
                if embedding_weights is not None:
                    self.lm_head.weight = embedding_weights
                else:
                    logger.warning("Could not find appropriate embeddings to tie with LM head, using random initialization.")
        except AttributeError as e:
            logger.warning(f"Could not tie LM head weights: {e}. Using random initialization.")
            logger.warning("Embedding structure may differ from expected - checking available attributes:")
            for attr_name in dir(self.model):
                if 'embed' in attr_name.lower():
                    logger.warning(f"  Found embedding-related attribute: {attr_name}")
        
        # Apply LoRA if configuration is provided
        if lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"Applied LoRA with config: {lora_config}")
        
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
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            raise AttributeError("Model doesn't support saving LoRA adapters")

    def load_lora_adapters(self, load_path):
        if hasattr(self.model, 'get_base_model'):
            base_model = self.model.get_base_model()
            self.model = PeftModel.from_pretrained(base_model, load_path)
        else:
            raise AttributeError("Model doesn't support loading LoRA adapters")


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
    
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
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
    
    # 1. Load config and base model outside the custom class
    config = EsmConfig.from_pretrained(model_name)
    base_model = EsmModel.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32
    )
    
    # 2. Pass the pre-loaded model into the custom class
    model = ESMWithConstraints(
        base_model=base_model,
        config=config,
        lora_config=lora_config
    )
    
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    
    return model, tokenizer
