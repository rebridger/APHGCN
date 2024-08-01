## APHGCN

Source code for "Adaptive-Propagating Heterophilous Graph Convolutional Network"

![APHGCN](https://github.com/rebridger/APHGCN/blob/main/framework.jpg?raw=true)

## Requirements
- torch==1.13.1
- numpy


## Run
```
python main.py
```

- For homophilous dataset (Cora, Citeseer, Minesweeper): data_split_mode = 'Num'

```
python main.py --data_split_mode = 'Num'
```

- For heterophilous datasets (Chameleon, Cornell, Film, Squirrel, Tesax, Wisconsin): data_split_mode = 'Ratio'

```
python main.py --data_split_mode = 'Ratio'
```