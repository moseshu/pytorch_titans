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
Step 60/26735 (0.22%) | Training Loss: 6.0509 | LM: 6.0509 | WM: 0.0000 | Gate: 0.000
Step 60/26735 | Validation Loss: 6.4431 | LM: 6.4431 | WM: 0.0000 | Gate: 0.000
Step 80/26735 (0.30%) | Training Loss: 6.1801 | LM: 6.1801 | WM: 0.0000 | Gate: 0.000
Step 80/26735 | Validation Loss: 6.0647 | LM: 6.0647 | WM: 0.0000 | Gate: 0.000
Step 100/26735 (0.37%) | Training Loss: 6.0434 | LM: 6.0434 | WM: 0.0000 | Gate: 0.000
Step 100/26735 | Validation Loss: 6.5956 | LM: 6.5956 | WM: 0.0000 | Gate: 0.000
Saving checkpoint to checkpoints/checkpoint-100...
Checkpoint saved.
Removed old checkpoint: checkpoints/checkpoint-100
Step 120/26735 (0.45%) | Training Loss: 6.3664 | LM: 6.3664 | WM: 0.0000 | Gate: 0.000
Step 120/26735 | Validation Loss: 5.8169 | LM: 5.8169 | WM: 0.0000 | Gate: 0.000
Step 140/26735 (0.52%) | Training Loss: 6.1446 | LM: 6.1446 | WM: 0.0000 | Gate: 0.000
Step 140/26735 | Validation Loss: 6.8763 | LM: 6.8763 | WM: 0.0000 | Gate: 0.000
Saving checkpoint to checkpoints/checkpoint-150...
Checkpoint saved.
Removed old checkpoint: checkpoints/checkpoint-150
Step 160/26735 (0.60%) | Training Loss: 6.3173 | LM: 6.3173 | WM: 0.0000 | Gate: 0.000
Step 160/26735 | Validation Loss: 5.8490 | LM: 5.8490 | WM: 0.0000 | Gate: 0.000
Step 180/26735 (0.67%) | Training Loss: 5.9880 | LM: 5.9880 | WM: 0.0000 | Gate: 0.000
Step 180/26735 | Validation Loss: 6.5561 | LM: 6.5561 | WM: 0.0000 | Gate: 0.000
Step 200/26735 (0.75%) | Training Loss: 5.8325 | LM: 5.8325 | WM: 0.0000 | Gate: 0.000
Step 200/26735 | Validation Loss: 5.8982 | LM: 5.8982 | WM: 0.0000 | Gate: 0.000
Saving checkpoint to checkpoints/checkpoint-200...
Checkpoint saved.
Removed old checkpoint: checkpoints/checkpoint-200
Step 220/26735 (0.82%) | Training Loss: 5.7310 | LM: 5.7310 | WM: 0.0000 | Gate: 0.000
Step 220/26735 | Validation Loss: 6.2626 | LM: 6.2626 | WM: 0.0000 | Gate: 0.000
Step 240/26735 (0.90%) | Training Loss: 6.2902 | LM: 6.2902 | WM: 0.0000 | Gate: 0.000
Step 240/26735 | Validation Loss: 6.1724 | LM: 6.1724 | WM: 0.0000 | Gate: 0.000
Saving checkpoint to checkpoints/checkpoint-250...
Checkpoint saved.
Removed old checkpoint: checkpoints/checkpoint-250
Step 260/26735 (0.97%) | Training Loss: 6.3755 | LM: 6.3755 | WM: 0.0000 | Gate: 0.000
Step 260/26735 | Validation Loss: 5.9196 | LM: 5.9196 | WM: 0.0000 | Gate: 0.000
Step 280/26735 (1.05%) | Training Loss: 6.1827 | LM: 6.1827 | WM: 0.0000 | Gate: 0.000
Step 280/26735 | Validation Loss: 6.0919 | LM: 6.0919 | WM: 0.0000 | Gate: 0.000
Step 300/26735 (1.12%) | Training Loss: 5.7777 | LM: 5.7777 | WM: 0.0000 | Gate: 0.000
Step 300/26735 | Validation Loss: 5.9194 | LM: 5.9194 | WM: 0.0000 | Gate: 0.000
Saving checkpoint to checkpoints/checkpoint-300...
....
Removed old checkpoint: checkpoints/checkpoint-4250
Step 4260/26735 (15.93%) | Training Loss: 1.5031 | LM: 1.5031 | WM: 0.0000 | Gate: 0.000
Step 4260/26735 | Validation Loss: 1.6541 | LM: 1.6541 | WM: 0.0000 | Gate: 0.000
Step 4280/26735 (16.01%) | Training Loss: 1.9408 | LM: 1.9408 | WM: 0.0000 | Gate: 0.000
Step 4280/26735 | Validation Loss: 1.4853 | LM: 1.4853 | WM: 0.0000 | Gate: 0.000
Step 4300/26735 (16.08%) | Training Loss: 2.0216 | LM: 2.0216 | WM: 0.0000 | Gate: 0.000
Step 4300/26735 | Validation Loss: 2.0111 | LM: 2.0111 | WM: 0.0000 | Gate: 0.000
Saving checkpoint to checkpoints/checkpoint-4300...
Checkpoint saved.
Removed old checkpoint: checkpoints/checkpoint-4300
Step 4320/26735 (16.16%) | Training Loss: 2.0457 | LM: 2.0457 | WM: 0.0000 | Gate: 0.000
Step 4320/26735 | Validation Loss: 1.5575 | LM: 1.5575 | WM: 0.0000 | Gate: 0.000
Step 4340/26735 (16.23%) | Training Loss: 1.7154 | LM: 1.7154 | WM: 0.0000 | Gate: 0.000
Step 4340/26735 | Validation Loss: 1.7498 | LM: 1.7498 | WM: 0.0000 | Gate: 0.000
Saving checkpoint to checkpoints/checkpoint-4350...
Checkpoint saved.
Removed old checkpoint: checkpoints/checkpoint-4350
Step 4360/26735 (16.31%) | Training Loss: 1.7499 | LM: 1.7499 | WM: 0.0000 | Gate: 0.000
Step 4360/26735 | Validation Loss: 1.9553 | LM: 1.9553 | WM: 0.0000 | Gate: 0.000
Step 4380/26735 (16.38%) | Training Loss: 1.7013 | LM: 1.7013 | WM: 0.0000 | Gate: 0.000
Step 4380/26735 | Validation Loss: 1.7171 | LM: 1.7171 | WM: 0.0000 | Gate: 0.000
Step 4400/26735 (16.46%) | Training Loss: 1.7304 | LM: 1.7304 | WM: 0.0000 | Gate: 0.000
Step 4400/26735 | Validation Loss: 1.4374 | LM: 1.4374 | WM: 0.0000 | Gate: 0.000
Saving checkpoint to checkpoints/checkpoint-4400...
Checkpoint saved.
Removed old checkpoint: checkpoints/checkpoint-4400
Step 4420/26735 (16.53%) | Training Loss: 1.6774 | LM: 1.6774 | WM: 0.0000 | Gate: 0.000
Step 4420/26735 | Validation Loss: 1.6489 | LM: 1.6489 | WM: 0.0000 | Gate: 0.000

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
