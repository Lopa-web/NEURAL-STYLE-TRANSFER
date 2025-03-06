import torch
import torch.optim as optim
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    size = max(image.size)
    if size > max_size:
        scale = max_size / size
        new_size = tuple([int(dim * scale) for dim in image.size])
        image = image.resize(new_size, Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

def imshow(tensor, title=None):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor.numpy().transpose(1, 2, 0)
    tensor = tensor * 0.229 + 0.485
    tensor = tensor.clip(0, 1)
    plt.imshow(tensor)
    if title:
        plt.title(title)
    plt.show()

vgg = models.vgg19(weights="IMAGENET1K_V1").features.eval()

def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        return torch.mean((x - self.target)**2)

class StyleLoss(nn.Module):
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        x = x.view(c, h * w)
        return torch.mm(x, x.t())
    def forward(self, x, target):
        return torch.mean((self.gram_matrix(x) - self.gram_matrix(target))**2)

content_image = load_image("content.jpg")
style_image = load_image("style.jpg")

layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
content_features = get_features(content_image, vgg, layers)
style_features = get_features(style_image, vgg, layers)

target_image = content_image.clone().requires_grad_(True)
optimizer = optim.LBFGS([target_image])
content_loss_fn = ContentLoss(content_features['19'])
style_loss_fn = StyleLoss()

iterations = 500
def closure():
    optimizer.zero_grad()
    target_features = get_features(target_image, vgg, layers)
    content_loss = content_loss_fn(target_features['19'])
    style_loss = sum(style_loss_fn(target_features[layer], style_features[layer]) for layer in layers)
    total_loss = content_loss + 1e6 * style_loss
    total_loss.backward(retain_graph=True)
    return total_loss

for i in range(iterations):
    total_loss = optimizer.step(closure)
    if i % 50 == 0:
        print(f"Iteration {i}: Loss = {total_loss.item()}")
        imshow(target_image, title=f"Iteration {i}")

output = target_image.squeeze(0).detach()
output = output * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
output = output.clamp(0, 1)
output = transforms.ToPILImage()(output)
output.save('styled_image.jpg')
