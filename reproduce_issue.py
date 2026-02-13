import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def test_temperature_stability():
    # Setup
    torch.manual_seed(42)
    
    # Mock embeddings
    batch_size = 8
    dim = 128
    
    # Initial random embeddings (normalized)
    pLM = F.normalize(torch.randn(batch_size, dim), dim=-1)
    pGNN = F.normalize(torch.randn(batch_size, dim), dim=-1)
    
    # Simulating a scenario where alignment is initially poor (random)
    # Average alignment of random vectors in high dim is near 0
    # But some might be negative.
    
    # Define Temperature as a learnable parameter
    temperature = nn.Parameter(torch.tensor(1.0))
    
    optimizer = torch.optim.SGD([temperature], lr=0.1)
    
    losses = []
    temps = []
    
    print(f"Initial Temperature: {temperature.item()}")
    
    # Training loop focusing only on temperature
    for epoch in range(100):
        # Calculate similarity
        sim_matrix = torch.matmul(pLM, pGNN.t()) * temperature
        
        labels = torch.arange(batch_size)
        loss = 0.5 * (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        temps.append(temperature.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Temp = {temperature.item():.4f}")
            
    # Now simulate what happens if temperature becomes negative
    print("\n--- Forcing Negative Temperature ---")
    temperature.data = torch.tensor(-1.0)
    
    # And suppose the model "learns" to align (pLM becomes closer to pGNN)
    # We simulate this by interpolating pLM towards pGNN
    
    alignment_losses = []
    steps = []
    
    for step in range(10):
        alpha = step / 10.0  # Interpolation factor
        # pLM moves towards pGNN
        current_pLM = F.normalize((1 - alpha) * pLM + alpha * pGNN, dim=-1)
        
        sim_matrix = torch.matmul(current_pLM, pGNN.t()) * temperature
        loss = 0.5 * (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels))
        
        alignment_losses.append(loss.item())
        steps.append(step)
        print(f"Alignment Step {step} (alpha={alpha:.1f}): Loss = {loss.item():.4f}")

if __name__ == "__main__":
    test_temperature_stability()
