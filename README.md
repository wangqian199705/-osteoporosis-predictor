# 基于机器学习的骨质疏松风险预测系统

本项目构建了一个交互式网页工具，用于预测用户是否存在骨质疏松风险。用户可输入多项健康指标（如BMI、年龄、血脂等）以及生活习惯（是否饮酒/吸烟等），系统将基于机器学习模型（Stacking集成模型）实时预测患病概率并给出医学建议。

---

## 项目特点

- 使用 NHANES 数据训练模型
- 支持特征标准化与缺失值填充
- 可视化预测结果（概率饼图）
- 医学建议自动生成
- 前端友好（支持“是/否”输入，而非 0/1）

---

## 快速开始

### 克隆项目

```bash
git clone https://github.com/your-username/osteoporosis-predictor.git
cd osteoporosis-predictor
