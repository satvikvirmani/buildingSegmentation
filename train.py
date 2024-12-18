import torch
from torch import nn
from model import model
from dataset import train_dataloader, test_dataloader
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


train_features, train_labels = next(iter(train_dataloader))
print(f"Input device: {train_features.device}, Model device: {next(model.parameters()).device}")

learning_rate = 1e-3
batch_size = 4

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     # Set the model to training mode - important for batch normalization and dropout layers
#     # Unnecessary in this situation but added for best practices
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         # Compute prediction and loss
#         X, y = X.to(device), y.to(device)

#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 10 == 0:
#             loss, current = loss.item(), batch * batch_size + len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model):
    model.eval()  # Set the model to evaluation mode
    predictions_to_show = 3  # Number of predictions to visualize
    shown_predictions = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch, (images, labels) in enumerate(dataloader):
            # Move data to the specified device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)

            # Make predictions (raw logits)
            logits = model(images)
            print(logits)
            # Apply sigmoid to convert logits to probabilities
            preds = torch.sigmoid(logits)
            print(preds)
            # Threshold probabilities to create binary masks
            #preds = (preds > 0.5).float()

            print(preds)

            # Display 3 predictions
            if shown_predictions < predictions_to_show:
                for i in range(images.size(0)):
                    if shown_predictions >= predictions_to_show:
                        break

                    # Prepare images and masks for display
                    img = images[i].permute(1, 2, 0).cpu().numpy()  # Convert [3, H, W] -> [H, W, C]
                    label = labels[i].squeeze().cpu().numpy()  # Convert [1, H, W] -> [H, W]
                    pred = preds[i].squeeze().cpu().numpy()  # Convert [1, H, W] -> [H, W]

                    # Display the original image, ground truth, and prediction
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(img)
                    plt.title("Image")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(label, cmap="gray")
                    plt.title("Ground Truth")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(pred, cmap="gray")
                    plt.title("Prediction")
                    plt.axis("off")

                    plt.show()

                    shown_predictions += 1

            if shown_predictions >= predictions_to_show:
                break

loss_fn = nn.BCELoss()  # Use binary cross entropy for segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    #test_loop(test_dataloader, model)
test_loop(test_dataloader, model)
print("Training complete!")