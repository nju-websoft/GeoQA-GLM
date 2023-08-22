# GeoQA-GLM


### 高考地理选择题
输入如下指令, 训练测试GLM回答选择题
```angular2html
python glm_mcqa.py --model_name $GLM模型路径$ --dataset_name $数据集名$ 
```
如
```angular2html
python glm_mcqa.py --model_name GeoQA-GLM/THUDM/glm-large-chinese/ --dataset_name GKMC 
```