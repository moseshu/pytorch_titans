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

```
