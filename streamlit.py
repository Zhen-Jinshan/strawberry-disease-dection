import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 设置页面配置
st.set_page_config(page_title="Strawberry Disease Detection", layout="wide")

# 自定义样式
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
    }
    .result-container {
        background-color: #ffc107;
        padding: 20px;
        border-radius: 5px;
    }
    .result-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #6c757d;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }
    .footer p {
        margin: 0;
        padding: 0;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# 加载 YOLOv8 模型
model = YOLO('/home/zjs/Downloads/DL-Based-Leaf-Disease-Detection-master/best.pt')  # 替换为你的best.pt路径

# 图像上传处理
def load_image(image_file):
    img = Image.open(image_file)
    return img

# 进行预测并返回结果
def predict(image):
    results = model(image)
    return results

def main():
    st.title("Strawberry Disease Detection")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Upload an image for object detection...")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploaded_file")

        if uploaded_file is not None:
            image = load_image(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.session_state['image'] = image

    if st.button("Predict"):
        if 'image' in st.session_state:
            # 使用 YOLOv8 进行预测
            results = predict(st.session_state['image'])

            with col2:
                # 显示预测结果
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown('<p class="result-header">Prediction Results:</p>', unsafe_allow_html=True)

                # 显示检测到的类别和置信度
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_name = r.names[cls_id]
                        st.markdown(f'<p>{class_name}: {confidence*100:.2f}%</p>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # 在图像上绘制检测框并显示
                fig, ax = plt.subplots()
                ax.imshow(np.array(st.session_state['image']))
                for r in results:
                    for box in r.boxes:
                        # 将 Tensor 从 GPU 移动到 CPU 并转换为 NumPy 数组
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
                        ax.add_patch(rect)
                        plt.text(x1, y1, f"{r.names[int(box.cls[0].cpu().numpy())]}: {box.conf[0].cpu().numpy():.2f}",
                                 color='white', fontsize=12, backgroundcolor="red")
                ax.axis('off')
                st.pyplot(fig)
        else:
            st.write("Please upload an image first.")

if __name__ == '__main__':
    main()

    # 添加自定义页脚
    st.markdown("""
    <div class="footer">
        <p>Enjoy the prediction and thanks for using.</p>
        <p>Thanks & regards,</p>
        <p>ZJS</p>
    </div>
    """, unsafe_allow_html=True)
