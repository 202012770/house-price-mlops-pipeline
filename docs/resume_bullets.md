# 简历项目文案

> 所有数字均可在仓库 `reports/` 中复现：`model_comparison.csv`、`ablation.csv`、`metrics.json`、`drift_report.json`。

## 中文版（推荐，3 条）

**端到端房价预测系统**｜个人项目｜Python · scikit-learn · CatBoost · Optuna · MLflow · FastAPI · Docker

- 设计并实现可复现特征流程：14 个领域组合特征 + 序数编码 + 折外目标编码 + Yeo-Johnson 幂变换，通过消融实验使 5 折交叉验证 RMSE 降低 **2.79%**，测试集 R² 由 0.8881 提升至 **0.9003**。
- 基于 Optuna 对 CatBoost、LightGBM、XGBoost、随机森林、Ridge 五类模型各执行 20 次试验的 5 折交叉验证调参，MLflow 记录全部参数与指标并注册最优模型；按交叉验证选型部署 CatBoost，留出集 **MAE $14,452、R² 0.874**。
- 以完整 Pipeline 交付服务：启动时单例加载、Pydantic v2 输入校验、批量推理、Prometheus 指标与结构化日志；实现 PSI + KS 数据漂移监控，同分布回放最大 PSI 0.119（未误报），人为漂移可准确识别 3 个特征。

## 中文版（4 条，适合一页简历空间充足时）

- 构建从数据校验、分层切分、特征工程、超参搜索、实验跟踪到在线服务的完整流水线，训练与推理共用同一 sklearn Pipeline，从结构上杜绝数据泄漏与线上线下不一致。
- 特征工程包含 14 个领域组合特征、18 列有序变量语义映射、23 列 One-Hot 及 Neighborhood 等高基数特征的折外目标编码；消融实验量化每一步贡献，CV RMSE 累计降低 2.79%。
- 完成 5 类模型 × 20 次 Optuna 试验的调参对比，MLflow 记录实验并注册模型；分析发现线性模型在极端样本上外推失效（测试 R² 0.28），树模型稳定在 0.87–0.89。
- 实现 FastAPI 服务与监控闭环：单例模型加载、批量接口、输入范围与枚举校验、Prometheus 指标、SQLite 预测日志、PSI 漂移报告；60 个测试 + CI（Ruff、mypy、pytest），覆盖率 62%。

## English version

**End-to-End House Price Prediction System** | Personal project | Python, scikit-learn, CatBoost, Optuna, MLflow, FastAPI, Docker

- Built a reproducible feature pipeline (14 domain interaction features, ordinal encoding, out-of-fold target encoding, Yeo-Johnson transform); an ablation study reduced 5-fold CV RMSE by **2.79%** and lifted holdout R² from 0.8881 to **0.9003**.
- Tuned CatBoost, LightGBM, XGBoost, Random Forest and Ridge with Optuna (20 trials each, 5-fold CV), tracked every run in MLflow and registered the selected model; deployed the CV winner (CatBoost) reaching **MAE $14,452 / R² 0.874** on the holdout set.
- Served the full pipeline with FastAPI: singleton model loading, Pydantic v2 validation, batch inference, Prometheus metrics and structured logging; added PSI/KS drift monitoring with a SQLite prediction log (max PSI 0.119 on same-distribution replay, 3 features flagged under injected drift).

---

## 面试时的三个"钩子"

1. **方法论**：选型只用交叉验证，测试集不参与决策——这是最容易被追问、也最能体现严谨性的点。
2. **可量化**：消融实验、模型对比、漂移阈值都有具体数字，且能指到仓库里的文件。
3. **踩过坑**：折外目标编码的泄漏、`MSSubClass` 浮点/字符串类别错位、可选字段缺失导致的漂移误报——每个都是真实的工程问题，比"我用了 XGBoost"有说服力。
