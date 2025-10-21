# ensemble_inference.py
import torch
import torch.nn.functional as F
from model import EfficientNetClassifier, ViTClassifier
from torchvision import transforms
from PIL import Image

def load_models(eff_path, vit_path, num_classes, device='cuda'):
    eff = EfficientNetClassifier(num_classes, model_name='efficientnet_b0').to(device)
    vit = ViTClassifier(num_classes, model_name='vit_base_patch16_224').to(device)
    eff.load_state_dict(torch.load(eff_path, map_location=device))
    vit.load_state_dict(torch.load(vit_path, map_location=device))
    eff.eval(); vit.eval()
    return eff, vit

def preprocess(img_path, img_size=224):
    tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return tf(img).unsqueeze(0)  # [1,C,H,W]

def predict_ensemble(img_tensor, models, device='cuda'):
    probs = []
    with torch.no_grad():
        for m in models:
            out = m(img_tensor.to(device))
            probs.append(F.softmax(out, dim=1))
    avg = torch.stack(probs).mean(dim=0)  # average probabilities
    pred = avg.argmax(dim=1).item()
    return pred, avg.cpu().numpy()

# usage:
# eff, vit = load_models('best_efficientnet.pth','best_vit.pth', num_classes=3)
# img = preprocess('some_leaf.jpg')
# pred, probs = predict_ensemble(img, [eff, vit])
