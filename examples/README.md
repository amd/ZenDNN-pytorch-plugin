# Examples
Given below are some examples for running inference for various models with zentorch. Note that you may need to install additional packages in your environment if not already present. Assuming you have installed zentorch plugin in your environment, you can install the rest of the packages by running:
```bash
pip install -r requirements.txt
```

## BERT
### Execute the following command to run inference for bert model:
```bash
python bert_example.py
```

### Output
Last hidden states shape: torch.Size([3, 339, 1024])

## DLRM
### Execute the following command to run inference for dlrm model:
```bash
python dlrm_example.py
```
### Output
```plain
AUC Score: 0.5
```


## Resnet
### Execute the following command to run inference for resnet model:
```bash
python resnet_example.py
```
### Output
```plain
plane, carpenter's plane, woodworking plane
```
