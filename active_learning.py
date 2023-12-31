import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 创建一个模拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0, random_state=42)
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.95, random_state=42)

# 初始训练集可能很小
initial_training_size = 50
X_train_initial = X_train[:initial_training_size]
y_train_initial = y_train[:initial_training_size]

# 剩下的是池子中等待标注的数据
X_pool = X_train[initial_training_size:]
y_pool = y_train[initial_training_size:]

# 建立一个简单的随机森林分类器作为我们的模型
model = RandomForestClassifier()

# 训练初始模型
model.fit(X_train_initial, y_train_initial)

# 迭代过程开始
n_queries = 100
for idx in range(n_queries):
    # 使用模型对池中的样本进行预测并计算不确定性，这里我们用预测概率来衡量
    probas = model.predict_proba(X_pool)
    # 选出最不确定的样本（probas接近0.5即不确定性高）
    uncertainty_index = np.argmax(np.max(probas, axis=1))

    # 从池中得到最不确定样本的特征
    uncertain_sample = X_pool[uncertainty_index].reshape(1, -1)

    # "专家"给出该样本的标签，这里我们直接用真实的y_pool来模拟
    uncertain_label = y_pool[uncertainty_index]

    # 将新数据加入到训练集中
    X_train_initial = np.append(X_train_initial, uncertain_sample, axis=0)
    y_train_initial = np.append(y_train_initial, [uncertain_label])

    # 从池中移除已经标注的样本
    X_pool = np.delete(X_pool, uncertainty_index, axis=0)
    y_pool = np.delete(y_pool, uncertainty_index)

    # 使用新的更大的训练集重新训练模型
    model.fit(X_train_initial, y_train_initial)

print("Active learning process completed.")

# 模型现在应该在预测新样本时更加准确。
