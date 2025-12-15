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
[Input]: Please explain the theory of relativity.
[Output]:
I do not have access to current information and the exact location of the first class. However, I can provide you with the list of available options for the previous class:

1. name: name, date, location, and location

2. location: name and location
3. location: location
4. location: name

3. location: location, location, location, and location

4. location: location, location, and location

5. location: location, location, and location
6. location: location, location, and location

7. location: the location of the location and the location, including the


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

#### 🧠 Latent Reasoning Block: Bridging Memory Retrieval and World Simulation

In the standard Titans architecture, Neural Memory is responsible for compressing historical context into weights and retrieving relevant information based on the current query. However, mere **retrieval** is not equivalent to **reasoning**. To realize the vision of a **World Model** (as proposed by Yann LeCun), where a model must not only remember the past but also simulate the future, we introduce the **Latent Reasoning Block (LRB)**.

##### 1. Motivation: System 1 vs. System 2
I conceptualize the Neural Memory in Titans as **System 1 (Intuition)**, which rapidly retrieves historical patterns (e.g., "objects generally fall down"). The Latent Reasoning Block acts as **System 2 (Reasoning)**, performing deep, non-linear computations in the latent space (e.g., "calculating the exact pixel coordinates of the falling object based on gravity and velocity").

##### 2. Mechanism
The LRB is positioned after the Neural Memory retrieval and before the attention mechanism. It integrates two distinct information streams:
1.  **Current Observation ($x_t$):** The input from the residual stream at the current time step.
2.  **Retrieved Dynamics ($r_t$):** The memory vector retrieved from the Neural Memory, encoding physical laws and long-term context.

The inference process is formalized as:

$$
h_{state} = x_t + r_t
$$
$$
h_{reasoned} = \mathcal{F}_{\theta}(h_{state})
$$
$$
y_t = \sigma(W_g [x_t; h_{reasoned}]) \odot h_{reasoned} + (1 - \sigma(W_g [x_t; h_{reasoned}])) \odot x_t
$$

Where $\mathcal{F}_{\theta}$ represents a deep Multi-Layer Perceptron (MLP) with SwiGLU activation, responsible for simulating state transitions. The gating mechanism ($\sigma$) allows the model to dynamically balance between raw observation (for static scenes) and reasoned simulation (for complex dynamics).

##### 3. Implications for World Modeling
For video generation and physical simulation tasks, the LRB addresses critical challenges:
*   **Physical Consistency:** While Neural Memory retains the *attributes* of objects (identity, material), the Reasoning Block calculates their *interactions* (collision, deformation).
*   **Temporal Coherence:** By explicitly processing the retrieved history alongside the current state, the module ensures that generated frames adhere to causal laws rather than just visual similarity.

---

### 📧 Contact & Correspondence

For any questions, discussions, or inquiries regarding the implementation of the Latent Reasoning Block or the Titans World Model architecture, please feel free to reach out:

**[Moses**: [moseshu25@gmail.com]
