import torch
import torch.nn as nn

# Input Img -> Hidden dim -> mean, std -> Parameterization Trick -> Decoder -> Output Image
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim = 200, z_dim = 20):
        super().__init__()

        # Encoder
        self.img_2hid = nn.Linear(input_dim, hidden_dim)
        self.hidden_mu = nn.Linear(hidden_dim, z_dim)
        self.hidden_sigma = nn.Linear(hidden_dim, z_dim)

        # Decoder
        self.z_2hid = nn.Linear(z_dim, hidden_dim)
        self.hidden_2img = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hidden_mu(h), self.hidden_sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))

        # We want to make sure that it returns the values that are between 0 and 1, because we will be normalising the images
        return torch.sigmoid(self.hidden_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.rand_like(sigma)
        z_new = mu + sigma * epsilon
        x_new = self.decode(z_new)

        return x_new, mu, sigma

if __name__ == "__main__":
    x = torch.randn(4, 28*28) # 28 * 28 = 784
    vae = VariationalAutoEncoder(input_dim = 28*28)
    x_new, mu, sigma = vae(x)
    print(f"Shape of X: {x_new.shape}")
    print(f"Shape of mu: {mu.shape}")
    print(f"Shape of sigma: {sigma.shape}")