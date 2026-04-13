import streamlit as st
import requests
import io
from PIL import Image
from deep_translator import GoogleTranslator

# --- 設定（クラウドのSecretsから読み込み） ---
HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 1. タイトルを小さく表示
st.subheader("🎨 FLUX.1 画像生成アプリ")

# 2. 日本語プロンプト入力欄
prompt_ja = st.text_area("プロンプト (日本語):", placeholder="例：着物を着た猫、浮世絵風")

# 3. NGプロンプト入力欄（復活）
negative_prompt_ja = st.text_area("NGプロンプト (日本語):", placeholder="例：低画質、文字、歪み", value="文字、ロゴ、低画質")

# 4. 裏側で翻訳処理（画面には出さない）
prompt_en = ""
negative_en = ""
if prompt_ja:
    prompt_en = GoogleTranslator(source='ja', target='en').translate(prompt_ja)
    if negative_prompt_ja:
        negative_en = GoogleTranslator(source='ja', target='en').translate(negative_prompt_ja)

# 5. 生成ボタン
if st.button("✨ 画像を生成"):
    if not prompt_ja:
        st.error("プロンプトを入力してください")
    else:
        with st.spinner("AIが画像を生成中..."):
            # FLUX.1へのリクエスト送信（NGプロンプトも含む）
            payload = {
                "inputs": prompt_en,
                "parameters": {"negative_prompt": negative_en}
            }
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                # 画像の表示（保存ボタンは削除）
                st.image(Image.open(io.BytesIO(response.content)), use_container_width=True)
            else:
                st.error("エラーが発生しました。時間を置いて試してください。")