import streamlit as st
import requests
import io
from PIL import Image
from deep_translator import GoogleTranslator

# --- 設定 ---
HF_TOKEN = st.secrets["HF_TOKEN"]

# 最新の schnell 用 URL (router経由)
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 1. アプリの見た目
st.set_page_config(page_title="FLUX Schnell", layout="centered")
st.subheader("⚡ FLUX.1 Schnell (爆速版)")

# 2. 入力エリア (デフォルトの文字をすべて削除しました)
prompt_base = st.text_area("① ベース設定 (例: cinematic, masterpiece):", 
                          value="", 
                          placeholder="空欄でもOKです")

prompt_add = st.text_area("② 追加指示 (何を描きたいか):", 
                         value="", 
                         placeholder="例: 青い空とひまわり畑")

negative_ja = st.text_area("NGプロンプト (入れたくないもの):", 
                          value="", 
                          placeholder="例: 文字、低画質")

# 3. 翻訳
combined_en = ""
negative_en = ""
if prompt_base or prompt_add:
    # 日本語を英語に翻訳
    full_prompt_ja = f"{prompt_base} {prompt_add}"
    combined_en = GoogleTranslator(source='ja', target='en').translate(full_prompt_ja)
    
    if negative_ja:
        negative_en = GoogleTranslator(source='ja', target='en').translate(negative_ja)

# 4. 生成ボタン
if st.button("✨ 爆速で生成"):
    if not (prompt_base or prompt_add):
        st.warning("何か指示を入力してください！")
    else:
        with st.spinner("数秒で描き上げます..."):
            payload = {
                "inputs": combined_en,
                "parameters": {
                    "negative_prompt": negative_en
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                st.image(Image.open(io.BytesIO(response.content)), use_container_width=True)
            elif response.status_code == 503:
                st.info("サーバー準備中。30秒ほど待って再試行してください。")
            else:
                st.error(f"エラーが発生しました (Code: {response.status_code})")
                st.write("詳細:", response.text)
