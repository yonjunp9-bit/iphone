import streamlit as st
import requests
import io
from PIL import Image
from deep_translator import GoogleTranslator

# --- 設定 ---
HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 1. タイトルを小さく表示
st.subheader("🎨 FLUX.1 カスタム生成アプリ")

# 2. OKプロンプト1：ベース設定（最初から文字を入れておけます）
prompt_base = st.text_area("① ベースプロンプト (固定設定など):", 
                          value="高画質、リアルな質感、ポートレート",
                          placeholder="例：実写風、アニメ調など")

# 3. OKプロンプト2：追加指示（その都度変える内容）
prompt_add = st.text_area("② 追加プロンプト (今の指示):", 
                         placeholder="例：公園で遊ぶ柴犬、笑顔で")

# 4. NGプロンプト
negative_prompt_ja = st.text_area("NGプロンプト (日本語):", 
                                 value="文字、ロゴ、低画質、奇妙な手",
                                 placeholder="例：ぼやけ、暗い")

# --- 翻訳と合体処理 ---
combined_en = ""
negative_en = ""

if prompt_base or prompt_add:
    # 2つを合体させる（読点を入れてつなぐ）
    full_prompt_ja = f"{prompt_base}、{prompt_add}"
    
    # 合体した日本語を翻訳
    combined_en = GoogleTranslator(source='ja', target='en').translate(full_prompt_ja)
    
    if negative_prompt_ja:
        negative_en = GoogleTranslator(source='ja', target='en').translate(negative_prompt_ja)

# 5. 生成ボタン
if st.button("✨ 画像を生成"):
    if not (prompt_base or prompt_add):
        st.error("プロンプトを入力してください")
    else:
        with st.spinner("AIが画像を生成中..."):
            payload = {
                "inputs": combined_en,
                "parameters": {"negative_prompt": negative_en}
            }
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                # 画像の表示
                st.image(Image.open(io.BytesIO(response.content)), use_container_width=True)
            else:
                st.error("エラーが発生しました。時間を置いて試してください。")