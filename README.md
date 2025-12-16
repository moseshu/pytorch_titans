# pytorch_titans
ref: [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch)
## download the dataset
[Dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
```python
dataset format
data = {"messages":[{"role":"user","content":xxx},{"role":"assistant","content":xxx},....]}
```
## launch the script
```python
sh launch_script.sh
```

### Loss
```python
Step 20 | Training Loss: 9.9755
Step 20 | Validation Loss: 9.8037
Step 40 | Training Loss: 8.1710
Step 40 | Validation Loss: 8.0368
Step 60 | Training Loss: 7.5819
Step 60 | Validation Loss: 7.5379
Step 80 | Training Loss: 7.5585
Step 80 | Validation Loss: 7.5315
Step 100 | Training Loss: 7.5574
Step 100 | Validation Loss: 7.5123
Step 120 | Training Loss: 7.4792
Step 120 | Validation Loss: 7.5018
Step 140 | Training Loss: 7.5203
Step 140 | Validation Loss: 7.4147
Step 160 | Training Loss: 7.4831
Step 160 | Validation Loss: 7.5462
Step 180 | Training Loss: 7.4838
Step 180 | Validation Loss: 7.4714
Step 200 | Training Loss: 7.3697
Step 200 | Validation Loss: 7.4896
....
Step 440 | Training Loss: 6.4870
Step 440 | Validation Loss: 6.5944
Checkpoint saved at step 450
Step 460 | Training Loss: 6.2658
Step 460 | Validation Loss: 6.3194
Step 480 | Training Loss: 6.3541
Step 480 | Validation Loss: 6.2988
Step 500 | Training Loss: 6.2601
Step 500 | Validation Loss: 6.1591

```


## Infrence

```python
total model params is 122M
max_length=256

[Input]: Write a poem about the ocean.
[Output]:
As the sun sets on the beach,
A canvas of flowers,
A canvas so deep,
A sight to behold.

The sun sets over the horizon,
A sight to see,
The stars twinkle above,
A sight that's so captivates.

The trees begin to sway,
The landscape is full of wonder,
The sky is a canvas of hues,
A canvas of colors, a sight to behold.

The landscape is breathtaking,
A breathtaking sight to see,
A perfect sight to behold.

The landscape is a sight to behold,
As if in awe of the beauty,
A masterpiece of nature
```

# Latent Reasoning for State Transition


### 1. Initialization (System 1 Intuition)

First, the module fuses the current sensory input with long-term memory to form a static context and an initial intuition. This passes through $L$ layers of non-linear transformations (MLP + GEGLU).

$$
\begin{aligned}
c_t &= x_t + m_t \quad &(\text{Static Context}) \\
h_t^{(0)} &= \mathcal{F}_{\text{Reasoning}}(c_t) \quad &(\text{Initial Intuition})
\end{aligned}
$$

Where $\mathcal{F}_{\text{Reasoning}}$ represents the stack of `Linear -> GEGLU -> Linear` residual layers defined in `self.reasoning_layers`.

---

### 2. Recurrent Reasoning with ACT (System 2 Loop)

The model enters an internal refinement loop. For each step $k$:

#### A. State Dynamics ($\mathcal{T}$)
Depending on your configuration (`use_inner_attention`), the state update mechanism $\mathcal{T}$ is either:

**Option 1: Inner Attention (Global Deduction)**
Allows tokens to attend to each other in the latent space during reasoning:

$$
\tilde{h}_t^{(k)} = h_t^{(k-1)} + \text{Attention}(\text{LayerNorm}(h_t^{(k-1)}))
$$

**Option 2: GRU Cell (Local Refinement)**
Updates each token's state independently based on the static context:

$$
\tilde{h}_t^{(k)} = \text{GRU}(\underbrace{c_t}_{\text{input}}, \underbrace{h_t^{(k-1)}}_{\text{hidden}})
$$

Then, normalization is applied:

$$
h_t^{(k)} = \text{LayerNorm}(\tilde{h}_t^{(k)})
$$

#### B. Adaptive Computation Time (ACT)
The model dynamically decides when to stop thinking using a halting probability

$p_t^{(k)}$.

$$
\begin{aligned}
p_t^{(k)} &= \sigma(W_h h_t^{(k)} + b_h) \quad &(\text{Halting Probability}) \\
R_t^{(k)} &= 1 - \sum_{j=1}^{k-1} p_t^{(j)} \quad &(\text{Remaining Budget}) \\
w_t^{(k)} &= \min(p_t^{(k)}, R_t^{(k)}) \quad &(\text{Effective Weight})
\end{aligned}
$$

The final "pondered" state 

$h_t^{\text{final}}$ is a weighted sum of all intermediate states:
$$
h_t^{\text{final}} = \sum_{k=1}^{K} w_t^{(k)} h_t^{(k)} + R_t^{(K)} h_t^{(K)}
$$

We also compute the **Ponder Cost** to penalize excessive thinking:

$$
\mathcal{L}_{\text{ACT}} = \frac{1}{N} \sum_{t=1}^N (K_t + R_t^{(K)})
$$

*(Where $K_t$ is the step where the cumulative probability crossed the threshold).*

---

### 3. World Modeling (Future Prediction)

To ensure the thought state understands environmental dynamics, we project $h_t^{\text{final}}$ to predict the latent states of future tokens $\tau$ steps ahead.

$$
\begin{aligned}
\hat{z}_{t+\tau} &= \text{MLP}_{\text{Predictor}}(h_t^{\text{final}}) \quad &(\text{Predicted Future}) \\
z_{t+\tau}^{\text{GT}} &= x_{t+\tau} \quad &(\text{Ground Truth Future})
\end{aligned}
$$

The auxiliary **Rollout Loss** (using Smooth L1) is:

$$
\mathcal{L}_{\text{WM}} = \sum_{\tau=1}^{\text{Horizon}} \text{SmoothL1}(\hat{z}_{t+\tau}, z_{t+\tau}^{\text{GT}})
$$

---

### 4. Gated Integration (Output)

Finally, a confidence gate determines how much of the "deliberate thought" ($h_t^{\text{final}}$) should update the original stream ($x_t$).

$$
\begin{aligned}
g_t &= \sigma \left( W_{g} \cdot [x_t ; h_t^{\text{final}}] + b_{g} \right) \quad &(\text{Confidence Gate}) \\
y_t &= g_t \odot h_t^{\text{final}} + (1 - g_t) \odot x_t \quad &(\text{Block Output})
\end{aligned}
$$

---

### Summary: Optimization Objective

The total training objective combines the Language Modeling loss with the World Model and ACT penalties:

$$
\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{WM}} \mathcal{L}_{\text{WM}} + \lambda_{\text{ACT}} \mathcal{L}_{\text{ACT}}
$$

This formulation forces the model to find a latent state $h^*$ that is **predictive of the future**, **computationally efficient**, and **useful for the immediate token prediction**.


---

### 📧 Contact & Correspondence

For any questions, discussions, or inquiries regarding the implementation of the Latent Reasoning Block or the Titans World Model architecture, please feel free to reach out:

**[Moses]**: [moseshu25@gmail.com]
