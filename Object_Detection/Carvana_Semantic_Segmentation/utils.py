import torch
import torchvision
import os


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer'])


def check_accuracy(loader, model, device='cuda'):
    """Eval model with pixel accuracy and dice score"""
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            x, y = batch['imgs'], batch['masks']
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct = (preds == y).sum()
            num_pixels = torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (  # element-wise multiply, both output white-pixel
                # element-wise addtion, each is 0 or 1
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels*100:.2f}")
    print(f'Dice score: {dice_score / len(loader)}')

    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    """Visualize eval image"""
    model.eval()

    for idx, batch in enumerate(loader):
        x, y = batch['imgs'], batch['masks']
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()
