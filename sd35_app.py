import streamlit as st
import requests
import io
from PIL import Image
from deep_translator import GoogleTranslator

# --- 設定 ---
HF_TOKEN = st.secrets["HF_TOKEN"]
# SD3.5 LargeのURL
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 1. タイトル
st.subheader("🚀 Stable Diffusion 3.5 アプリ")

# 2. OKプロンプト1：ベース
prompt_base = st.text_area("① ベースプロンプト:", 
                          value="photorealistic, highly detailed, 8k",
                          placeholder="例：実写風、ファンタジー風など")

# 3. OKプロンプト2：追加指示
prompt_add = st.text_area("② 追加プロンプト:", 
                         placeholder="例：宇宙を泳ぐクジラ、幻想的")

# 4. NGプロンプト
negative_prompt_ja = st.text_area("NGプロンプト (日本語):", 
                                 value="文字、ロゴ、低画質、奇妙な手",
                                 placeholder="例：ぼやけ")

# --- 翻訳と合体 ---
combined_en = ""
negative_en = ""

if prompt_base or prompt_add:
    full_prompt_ja = f"{prompt_base}、{prompt_add}"
    combined_en = GoogleTranslator(source='ja', target='en').translate(full_prompt_ja)
    if negative_prompt_ja:
        negative_en = GoogleTranslator(source='ja', target='en').translate(negative_prompt_ja)

# 5. 生成ボタン
if st.button("✨ SD3.5で生成"):
    if not (prompt_base or prompt_add):
        st.error("プロンプトを入力してください")
    else:
        with st.spinner("SD3.5が考え中... (初回は時間がかかります)"):
            payload = {
                "inputs": combined_en,
                "parameters": {"negative_prompt": negative_en}
            }
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                st.image(Image.open(io.BytesIO(response.content)), width='stretch')
            elif response.status_code == 503:
                st.info("モデルを起動中です。1分ほど待ってからもう一度押してください。")
            else:
                st.error(f"エラーが発生しました (Code: {response.status_code})")
