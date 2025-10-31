from transformers import EsmModel

model_name = "facebook/esm2_t33_650M_UR50D"
try:
    base_model = EsmModel.from_pretrained(model_name)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")