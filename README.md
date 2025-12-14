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
