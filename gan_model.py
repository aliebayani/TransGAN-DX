import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(X_train, input_dim, hidden_dim, batch_size, epochs):
    generator = Generator(input_dim=100, hidden_dim=hidden_dim, output_dim=input_dim)
    discriminator = Discriminator(input_dim=input_dim, hidden_dim=hidden_dim)

    gen_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    loss_function = nn.BCELoss()

    for epoch in range(epochs):
        real_data = torch.tensor(X_train.sample(batch_size, random_state=42).values.astype(np.float32))
        labels_real = torch.ones(batch_size, 1)
        labels_fake = torch.zeros(batch_size, 1)

        # Discriminator training
        disc_optimizer.zero_grad()
        output_real = discriminator(real_data)
        loss_real = loss_function(output_real, labels_real)

        noise = torch.randn(batch_size, 100)
        fake_data = generator(noise)
        output_fake = discriminator(fake_data)
        loss_fake = loss_function(output_fake, labels_fake)

        disc_loss = loss_real + loss_fake
        disc_loss.backward()
        disc_optimizer.step()

        # Generator training
        gen_optimizer.zero_grad()
        noise = torch.randn(batch_size, 100)
        fake_data = generator(noise)
        output_fake = discriminator(fake_data)
        gen_loss = loss_function(output_fake, labels_real)
        gen_loss.backward()
        gen_optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] - Discriminator Loss: {disc_loss.item()}, Generator Loss: {gen_loss.item()}")
