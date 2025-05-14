# Reinforcement_hw4

# HW4-1 Understanding Report

以下報告聚焦於**Basic DQN 實作於簡易環境**（4×4 Gridworld static mode）與**Experience Replay Buffer**兩大重點，說明其原理、流程與核心程式架構。

---

## 1. Basic DQN Implementation for an Easy Environment

### 1.1 問題設定與環境

* **環境**：4×4 Gridworld，採用 static mode，所有目標（Goal）、陷阱（Pit）、牆壁（Wall）位置不變。
* **狀態維度**：將 4×4×4 的 one-hot 表示攤平成長度 64 的向量。
* **動作空間**：4 種方向（up, down, left, right），對應輸出層寬度為 4。

### 1.2 網路架構

```python
model = Sequential([
    Dense(150, activation='relu', input_shape=(64,)),
    Dense(100, activation='relu'),
    Dense(4)  # Q-value for each action
])
```

* **輸入層**：64 個節點；
* **隱藏層**：兩層全連接，分別為 150、100 個隱藏神經元；
* **輸出層**：4 個節點，對應每個動作的 Q-value。

### 1.3 訓練流程 (伪代码)

```text
for episode in range(max_episodes):
    state = env.reset()                # 獲得初始狀態向量 s
    done  = False
    while not done:
        # 1) ε-greedy 選擇動作
        if random() < epsilon:
            a = random_action()
        else:
            q_values = model.predict(s)
            a = argmax(q_values)

        # 2) 執行動作，取得 r, s', done
        s_next, r, done = env.step(a)

        # 3) 立即更新 Q-network
        #    TD target: Y = r + γ · max_a' Q(s', a')
        q_next = model.predict(s_next)
        Y      = r + (1-done)*γ * max(q_next)
        #    損失: L = (Q(s,a) - Y)^2
        loss = MSE(model(s)[a], Y)
        #    反向傳播
        optimizer.minimize(loss)

        s = s_next
    # ε 漸降至 ε_min
```

* **重點說明**：此版本尚未引入 Target Network，直接以當前網路輸出計算 $\max Q(s',a')$，適用於簡易、小規模環境的初步測試。

---

## 2. Experience Replay Buffer

### 2.1 目的與動機

1. **去相關化**：隨機抽取多筆過去經驗，打破「連續時間序列」間的強相關性，提升收斂穩定度。
2. **樣本再利用**：同一筆經驗在不同時間可被重複使用，提升樣本效率（sample efficiency）。

### 2.2 結構與操作

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        # 分解成五個陣列，分別轉為 tensor
        return s_batch, a_batch, r_batch, s_next_batch, done_batch

    def __len__(self):
        return len(self.buf)
```

* **儲存**：每一步 $(s,a,r,s',done)$ 推入 `deque`，採用先進先出策略；
* **抽樣**：當 `len(buffer) >= batch_size` 時，隨機抽取批次，用於一次更新；
* **Batch 更新**：

  ```text
  Y_batch = r_batch + (1 - done_batch) * γ * max_a' Q(s'_batch, a')
  L = MSE(Q(s_batch, a_batch), Y_batch)
  ```

### 2.3 實作要點

* 保證抽樣後的 `state` 張量形狀為 $(B,64)$，動作 `a` 形狀為 $(B,1)$。
* 更新前先 `optimizer.zero_grad()`，再 `loss.backward()`、`optimizer.step()`。
* 調整 `capacity`、`batch_size` 可影響訓練穩定度與效能。

---
## 3.成果展示
![43569688-83e4-4076-93c0-d20bf08946b6](https://github.com/user-attachments/assets/fc509607-4c30-4e73-9583-7a735d17c106)

---

## 4. 小結

* **Basic DQN** 為理解 Q-learning 與深度網路結合的基礎，流程簡單明瞭；
* **Experience Replay** 是 DQN 穩定訓練的關鍵，通過隨機抽樣與樣本再利用，大幅提升收斂速率與穩定度。

以上即為 HW4-1 之核心理解報告，建議實作後透過變動超參數（如 $\gamma$、batch size、network depth）觀察對學習曲線的影響。



# **HW4-2 Understanding Report**

本報告聚焦於 **Double DQN** 與 **Dueling DQN** 兩種演算法，並說明它們如何在基本 DQN 的基礎上改良、減少過度估計與加速收斂。

---

## 1. 基本 DQN 的局限性：過度估計偏誤 (Overestimation Bias)

* **Q-learning 目標**：
  $Y = r + \gamma \max_{a'} Q(s',a';\theta)$
  DQN 直接用同一張網路 $Q(\cdot;\theta)$ 來選擇與評估下一步的動作，導致對 **高估值** 的偏誤，影響學習穩定性。

* **現象**：在具有隨機性或噪聲的環境中，連續取最大值往往會放大隨機誤差，使得 Q 值普遍偏高，不易收斂。

---

## 2. Double DQN：分離「選擇」與「評估」網路

### 2.1 核心機制

```python
# online-net 選最大動作
next_actions = argmax_a Q(s'; θ_online)
# target-net 評估該動作的Q值
next_q = Q(s'; θ_target)[next_actions]
# TD 目標值
y = r + γ · next_q
```

* **選擇** (action selection) 與 **評估** (action evaluation) 分屬不同的網路參數， $θ_{online}$  與 $θ_{target}$ 互相獨立。

### 2.2 改良效益

1. **減少高估**：分開更新機制避免同網路同時選擇與評估最大動作，抑制隨機噪聲造成的偏差。
2. **穩定性提升**：搭配固定周期複製的 Target Network，可進一步平滑學習曲線。

---

## 3. Dueling DQN：劃分價值與優勢函數

### 3.1 網路架構

將最後一層分為兩條支路：

```text
           ┌─────────┐
      f(s) │ Feature │
           └─────────┘
             /      \
            /        \
       V(s)          A(s,a)
        │               │
        └───────┬───────┘
                ▼
      Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
```

* **Value-stream** $V(s)$：估計狀態整體價值
* **Advantage-stream** $A(s,a)$：估計各動作相對於平均水平的增益
* **融合**： $Q = V + A - 	frac{1}{|A|}\sum A$ ，確保唯一性

### 3.2 改良效益

1. **分解學習**：在一些狀態下動作差異微小（例如無行動區域），可獨立強化對狀態價值的學習，不受動作噪聲干擾。
2. **加速收斂**：提升網路對狀態與動作價值的結構化表達，特別在高維度或大動作空間時，能更快辨識「好/壞」狀態。

---

## 4. 實驗比較與結論

| 演算法         | 過度估計偏誤           | 收斂速度 | 最終表現     |
| ----------- | ---------------- | ---- | -------- |
| Basic DQN   | 高                | 中    | 中等偏上     |
| Double DQN  | 低                | 較快   | 穩定提升     |
| Dueling DQN | 中（可進一步搭配 Double） | 最快   | 最佳（明顯優勢） |

* **Double DQN** 通過分離選擇／評估顯著降低估計偏誤，訓練更穩定；
* **Dueling DQN** 對狀態價值進行明確建模，加速策略辨識，尤其在複雜或噪聲較大的場景中效果更佳；
* 可將兩者結合為 **Double Dueling DQN**，同時取兩種機制優勢，取得最出色的訓練成效。

---
![89e3fc90-5eb1-44bb-97b9-4b74e0f6a2b2](https://github.com/user-attachments/assets/b7604b08-e92c-422d-835d-1ee6826e208e)
![dcae87aa-aec6-44b3-b93a-6f00dc872b7a](https://github.com/user-attachments/assets/f0cc4611-9dbf-48a5-a23d-159463a7e803)

> **參考**：Hasselt et al. (2016) *Deep Reinforcement Learning with Double Q-learning*；
> Wang et al. (2016) *Dueling Network Architectures for Deep Reinforcement Learning*。
# HW4-3 理解報告

本報告聚焦於**增強型 DQN**在**random mode**下的實作改良，說明如何透過框架轉換與訓練優化技巧，提升訓練的穩定性、收斂速度及策略泛化能力。

---

## 1. 隨機模式 (Random Mode) 概述

* **環境**：4×4 Gridworld，每回合就隨機佈置玩家、目標、陷阱和牆壁。
* **挑戰**：高度隨機性導致觀測空間及動態不穩定，需要策略能適應不同佈局。
* **目標**：在 random mode 中訓練出穩健的 DQN，並將原始 PyTorch 實作轉為業界框架，同時引入訓練技巧進行優化。

---

## 2. 框架轉換

### 2.1 PyTorch Lightning 實作亮點

* **結構分離**：Lightning 自動管理訓練迴圈、GPU/TPU 設備切換與日誌紀錄，使用者只需關注 `LightningModule` 裡的核心邏輯。
* **Hook 函式**：

  * `training_step`：定義一次前向推理、損失計算與梯度回傳。
  * `configure_optimizers`：設定 optimizer 與 learning rate scheduler，自動調用。
* **Target Network 同步**：於 `global_step` 滿足條件時自動複製，有助降低手動錯誤。

### 2.2 Keras / TensorFlow 2.x 實作亮點

* **Eager Execution**：使用 `tf.GradientTape` 進行前向與反向傳播，程式可讀性佳。
* **手動迴圈 Control**：訓練迴圈與 PyTorch 版本相似，但整合 Keras API 定義網路與優化步驟。
* **模型建構**：採用 `tf.keras.Sequential` 及 Functional API，快速堆疊 Dense Layer。

---

## 3. 訓練優化技巧

### 3.1 梯度裁剪 (Gradient Clipping)

* **目的**：防止 TD-error 突然變大導致梯度爆炸。
* **Lightning**：在 `Trainer(gradient_clip_val= X)` 中設定。
* **Keras**：於 optimizer 中設置 `clipnorm` 或在 `apply_gradients` 時裁剪。
* **效果**：減少訓練初期損失震盪，提高穩定性。

### 3.2 學習率調度 (Learning Rate Scheduling)

* **目的**：大初始學習率加快早期探索，衰減後微調收斂。
* **Lightning**：於 `configure_optimizers` 回傳 `StepLR`、`CosineAnnealingLR` 等 Scheduler。
* **Keras**：使用 `tf.keras.optimizers.schedules` 或 Callback（如 `LearningRateScheduler`）。
* **效果**：減少搖擺，提升最終策略性能。

### 3.3 優先回放 (Prioritized Replay)

* **目的**：以 TD-error 為依據，讓有用的經驗被更頻繁抽樣，加速學習。
* **方法**：維護每筆 transition 的抽樣機率與重要性採樣權重。
* **效果**：強化稀有或關鍵轉移，提高策略在隨機佈局下的適應能力。

---

## 4. 實作重點對比

| 組件            | Basic DQN   | 增強型 DQN (Lightning/Keras)           |
| ------------- | ----------- | ----------------------------------- |
| 訓練迴圈          | 手動 for-loop | Lightning Hooks / `GradientTape` 管理 |
| Target Net 同步 | 手動複製權重      | `global_step % N == 0` 自動複製         |
| 日誌紀錄          | `print()`   | TensorBoard / Lightning 日誌          |
| LR 調度         | 手動呼叫        | 框架內建 Scheduler API                  |
| 梯度裁剪          | 無           | API 一行設定                            |
| 回放緩衝區抽樣方法     | 均勻抽樣        | 支援優先回放                              |

---

## 5. 效能與穩定性提升

* **訓練穩定性**：損失曲線振盪減少，梯度爆炸機率下降。
* **收斂速度**：初期高 lr + scheduler 衰減可快速降損並平滑收斂。
* **泛化能力**：優先回放將關鍵 transition 強化訓練，使策略更能適應多樣化佈局。
![52e0d272-f7f6-4afe-9997-86e68402f3a6](https://github.com/user-attachments/assets/1bb342a8-b78e-43d1-bdae-b850cbd49925)

---

## 6. 小結

透過現代深度學習框架與訓練優化技巧，增強型 DQN 不僅簡化了程式架構，也藉由梯度裁剪、學習率調度與優先回放等方法，大幅提升 random mode 下的訓練效率與策略質量。這些改良有效緩解了基本 DQN 的收斂不穩與過度估計問題，並為大規模或高噪音場景下的強化學習提供更可靠的解決方案。

---

> **參考文獻**：
>
> * Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015.
> * Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016.
> * Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning*, ICML 2016.
> * Van der Plas et al., *PyTorch Lightning: A lightweight PyTorch wrapper for high-performance research*, 2020.

