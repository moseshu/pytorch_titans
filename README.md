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


#### 🧠 Latent Reasoning Block: Integrating System 2 Thinking into Long-Context LLMs

In standard sequence modeling, retrieving past information is often insufficient for complex tasks requiring logical deduction. While the **Neural Memory** in Titans efficiently compresses thousands of tokens into synaptic weights—acting as a robust retrieval mechanism—it lacks the capacity for deep processing of that retrieved information. To bridge this gap, we introduce the **Latent Reasoning Block (LRB)**.

##### 1. Motivation: From Retrieval to Reasoning
We conceptualize the Neural Memory retrieval as **System 1 (Intuition)**, providing immediate access to historical context (e.g., retrieving a variable definition from the beginning of the code). The Latent Reasoning Block serves as **System 2 (Reasoning)**. It performs multi-step, non-linear processing in the latent space to derive logical implications from the retrieved memory before generating the next token.

##### 2. Mechanism
Positioned after the Neural Memory retrieval and before the attention mechanism, the LRB processes two distinct streams:
1.  **Current Input ($x_t$):** The embedding of the current token.
2.  **Retrieved Context ($r_t$):** The memory vector extracted from Neural Memory, encoding long-term dependencies and context.

The reasoning process is formalized as:

$$
h_{state} = x_t + r_t
$$
$$
h_{reasoned} = \mathcal{F}_{\theta}(h_{state})
$$
$$
y_t = \sigma(W_g [x_t; h_{reasoned}]) \odot h_{reasoned} + (1 - \sigma(W_g [x_t; h_{reasoned}])) \odot x_t
$$

Here, $\mathcal{F}_{\theta}$ represents a deep MLP with SwiGLU activation. This module acts as an "implicit reasoning engine," performing logical deductions in vector space without generating explicit chain-of-thought tokens.

##### 3. Impact on Language Modeling
For text generation and understanding tasks, the LRB offers significant advantages:
*   **Complex Inference:** It enables the model to synthesize information from distant parts of the text (retrieved by Memory) to answer questions requiring deductive reasoning.
*   **Global Consistency:** In long-form generation (e.g., novels or codebases), the LRB ensures that the current generation aligns logically with long-term constraints established earlier in the context.
*   **Computational Depth:** By allocating dedicated parameters for processing retrieved memories, the model can separate "remembering facts" from "applying logic," leading to more robust generalization.


---

### 📧 Contact & Correspondence

For any questions, discussions, or inquiries regarding the implementation of the Latent Reasoning Block or the Titans World Model architecture, please feel free to reach out:

**[Moses]**: [moseshu25@gmail.com]
