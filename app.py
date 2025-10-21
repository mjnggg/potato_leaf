import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from model import HybridEfficientTransformer  # ✅ đổi đúng class trong model.py

# ===== CONFIG =====
MODEL_PATH = "best_hybrid.pth"
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']  # ⚠️ sửa đúng nhãn thật của bạn

# ===== Load model =====
@st.cache_resource
def load_model(weights_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridEfficientTransformer(num_classes=num_classes, pretrained=False).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device

model, device = load_model(MODEL_PATH, len(CLASS_NAMES))

# ===== Giao diện Streamlit =====
st.title("🍃 Nhận diện bệnh lá khoai tây (EfficientNet + Transformer Hybrid)")

uploaded_file = st.file_uploader("📤 Tải ảnh lá khoai tây lên để kiểm tra:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = CLASS_NAMES[probs.argmax(dim=1).item()]
        confidence = probs.max().item() * 100

    st.success(f"✅ Kết quả dự đoán: **{pred_class}** ({confidence:.2f}% tin cậy)")
