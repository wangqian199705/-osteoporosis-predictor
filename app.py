import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="骨质疏松预测系统", layout="wide")
st.markdown("""
# 🦴 基于机器学习的骨质疏松预测系统
> 请输入以下健康信息，系统将为您预测是否存在骨质疏松风险，并提供医学建议。
""")

# -------------------- 加载模型和处理器 --------------------
basedir = os.path.dirname(__file__)
model = joblib.load(os.path.join(basedir, "stacked_model_new.pkl"))
scaler = joblib.load(os.path.join(basedir, "scaler_new.pkl"))
imputer = joblib.load(os.path.join(basedir, "imputer_new.pkl"))

with open(os.path.join(basedir, "feature_list_new.txt")) as f:
    feature_list = f.read().splitlines()

# -------------------- 预测函数 --------------------
def predict_op(input_df):
    X_input = input_df[feature_list]
    X_imputed = imputer.transform(X_input)
    X_scaled = scaler.transform(X_imputed)
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    return prediction, proba

# -------------------- 页面布局 --------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("📝 输入健康信息")
    user_input = {}
    float_features = {
        "BMI": "体质指数 (kg/m²)",
        "Hba1c": "糖化血红蛋白 (%)",
        "HDL": "高密度脂蛋白 (mmol/L)",
        "Sodium": "血钠浓度 (mmol/L)",
        "uric acid": "尿酸 (μmol/L)",
        "creatinine": "肌酐 (μmol/L)",
        "CHO": "总胆固醇 (mmol/L)",
        "Age": "年龄 (岁)"
    }
    binary_features = {
        "Gender": "性别 (0=女, 1=男)",
        "Activity": "活动 (0=少, 1=多)",
        "Alchol": "是否饮酒 (0=否, 1=是)",
        "Sleep-disorder": "是否有睡眠障碍 (0=否, 1=是)",
        "smoking": "是否吸烟 (0=否, 1=是)"
    }

    for key, label in float_features.items():
        user_input[key] = st.number_input(label, value=0.0)
    for key, label in binary_features.items():
        user_input[key] = st.selectbox(label, [0, 1], index=0)

    input_df = pd.DataFrame([user_input])

    if st.button("🔍 立即预测"):
        pred, prob = predict_op(input_df)

        with right:
            st.subheader("📊 预测结果")
            result = "患有骨质疏松" if pred == 1 else "未患骨质疏松"
            color = "red" if pred == 1 else "green"

            fig = go.Figure(data=[
                go.Pie(labels=["患骨质疏松概率", "未患概率"],
                       values=[prob, 1 - prob],
                       hole=0.4,
                       marker_colors=["#EF553B", "#00CC96"])
            ])
            fig.update_layout(title_text=f"预测概率：{prob:.1%} ({result})")

            st.plotly_chart(fig)

            # 医学建议
            if pred == 1:
                st.markdown("""
                ### 🩺 医学建议：
                - 您可能存在骨质疏松风险，建议尽早就医。
                - 可进一步检查骨密度（DEXA 检查）。
                - 调整饮食结构，增加钙摄入，避免久坐不动。
                """)
            else:
                st.markdown("""
                ### ✅ 医学建议：
                - 当前未显示骨质疏松风险。
                - 请保持良好生活方式：多运动、均衡饮食、定期体检。
                """)

    else:
        with right:
            st.subheader("📊 预测结果将在此处显示")
            st.markdown("请在左侧输入完信息后点击“立即预测”按钮")