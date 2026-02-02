
import torch
from models.gearnet_model import GearNetFromCoordinates

def test_gearnet():
    """
    Tests the GearNetFromCoordinates model with a small graph to ensure it does not hang.
    """
    print("Testing GearNetFromCoordinates with a small graph...")

    # Create a small graph with 5 nodes
    batch_size = 1
    seq_len = 5
    n_coords = torch.randn(batch_size, seq_len, 3)
    ca_coords = torch.randn(batch_size, seq_len, 3)
    c_coords = torch.randn(batch_size, seq_len, 3)

    # Create a GearNetFromCoordinates model
    model = GearNetFromCoordinates(hidden_dim=128, freeze=True)

    # Run the forward pass
    try:
        print("Running forward pass...")
        embeddings = model(n_coords, ca_coords, c_coords)
        print("Forward pass completed successfully.")
        print("Output shape:", embeddings.shape)
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")

if __name__ == "__main__":
    test_gearnet()
