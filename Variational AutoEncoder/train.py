import torch
import torchvision.datasets as datasets
from tqdm import tqdm
import torch.nn as nn
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data  import DataLoader

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 28 * 28
HIDDEN_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4

# Loading Dataset
dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = True)
model = VariationalAutoEncoder(INPUT_DIM, HIDDEN_DIM, Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LR_RATE)
# For reconstruction
loss_fn = nn.BCELoss(reduction = "sum")

# For training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:

        # Forward pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_new, mu, sigma = model(x)

        # Compute Loss
        reconstruction_loss = loss_fn(x_new, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # Backpropagation
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss = loss.item())

def interference(digit, num_examples=1):
    images = []
    
    # Collect a single image of the specified digit
    for x, y in dataset:
        if y == digit:  # Match the digit
            images.append(x.view(1, -1))  # Flatten the image
            break  # Stop after finding one example for the target digit

    if not images:
        print(f"No examples found for digit {digit}")
        return

    # Move the selected image to the same device as the model
    image = images[0].to(DEVICE)

    # Encode and decode to generate new samples
    with torch.no_grad():
        mu, sigma = model.encode(image)  # Get encoding
        for example in range(num_examples):
            epsilon = torch.randn_like(sigma).to(DEVICE)  # Random noise on the same device
            z = mu + sigma * epsilon  # Latent space sampling
            out = model.decode(z).to("cpu")  # Decode back to image
            out = out.view(-1, 1, 28, 28)  # Reshape to image dimensions

            # Save generated images with unique filenames
            save_image(out, f"generated_digit_{digit}_example_{example}.png")

for idx in range(10):
    interference(idx, num_examples = 1)