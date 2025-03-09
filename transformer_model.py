import torch
import torch.nn as nn
import torch.optim as optim

# Define the Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(SimpleTransformer, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(model_dim, output_dim)
    
    def forward(self, src):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        output = self.fc(output.mean(dim=1))
        return output

# Hyperparameters
input_dim = 10000  # Vocabulary size
model_dim = 512    # Embedding dimension
num_heads = 8      # Number of attention heads
num_layers = 6     # Number of transformer layers
output_dim = 10    # Number of output classes

# Instantiate the model
model = SimpleTransformer(input_dim, model_dim, num_heads, num_layers, output_dim)

# Print the model architecture
print(model)

# Sample data
src = torch.randint(0, input_dim, (32, 20))  # (batch_size, sequence_length)
labels = torch.randint(0, output_dim, (32,))  # (batch_size,)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(src)
    
    # Compute loss
    loss = criterion(output, labels)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_src = torch.randint(0, input_dim, (1, 20))  # (batch_size, sequence_length)
    output = model(test_src)
    predicted_class = torch.argmax(output, dim=1)
    print(f'Predicted class: {predicted_class.item()}')
