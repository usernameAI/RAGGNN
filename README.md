# RAGGNN
This is the official source code for our PR&AI (模式识别与人工智能) 2024 Paper

"Temporal Features Enhanced Graph Neural Network for Session-based Recommendation System in E-commerce"

"电商场景下时序特征增强的图神经网络会话推荐系统"

# Abstract
Most of the existing research on graph neural network-based session recommender systems focuses on capturing the contextual relationships of items in the session graph, but few address temporal relationships. However, both relationships are important for precise recommendations in e-commerce scenarios. This study aims to learn the contextual features and temporal relationships between products in anonymous user's clickstream in sessions. Bidirectional long and short-term memory networks and gated graph neural networks are adopted to capture the features, respectively. Meanwhile, the attention mechanism is adopted to denoising. Finally, adaptive feature fusion between both features is achieved based on the gating mechanism. The proposed model is evaluated on three publicly available benchmark datasets and outperforms fourteen baseline models in four metrics: precision, hit rate, mean reciprocal ranking, and normalized discounted cumulative gain. Cold start testing demonstrates that the proposed model can still effectively recommend with fewer training data. Ablation studies confirm the validity of each module. Additionally, the model is less affected by some hyperparameters and shows strong robustness. Furthermore, the study comparatively analyses the different user interests learned by the model based on different features through visual case presentations.

# Dataset
We provide one dataset: Diginetica. 

The Yoochoose dataset can be found at: http://2015.recsyschallenge.com/challege.html

# Example to run the codes
1. Install RecBole: `pip install recbole`

2. RecBole-GNN can be found at: https://github.com/RUCAIBox/RecBole-GNN

3. Run RAGGNN

`python run.py`
