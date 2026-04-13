import streamlit as st
import requests
import io
from PIL import Image
from deep_translator import GoogleTranslator

# --- 設定 ---
HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

st.title("🎨 FLUX.1 iPhone対応版")

prompt_ja = st.text_area("プロンプト (日本語):", placeholder="例：猫、サイバーパンク")

if prompt_ja:
    prompt_en = GoogleTranslator(source='ja', target='en').translate(prompt_ja)
    st.subheader("📝 翻訳結果")
    st.info(prompt_en)

if st.button("✨ 画像を生成"):
    if not prompt_ja:
        st.error("入力してください")
    else:
        with st.spinner("生成中..."):
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt_en})
            if response.status_code == 200:
                # 画像を表示
                img_data = response.content
                img = Image.open(io.BytesIO(img_data))
                st.image(img, caption="生成された画像")

                # --- 保存（ダウンロード）ボタンの設置 ---
                st.download_button(
                    label="💾 画像を保存する",
                    data=img_data,
                    file_name="generated_image.png",
                    mime="image/png"
                )
                st.success("ボタンを押すとiPhoneに保存できます！")
            else:
                st.error("エラーが発生しました。")