# Linear Least Squares (LLS) III

## Tehran House Price

<img src="https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/d9500d85-9f5b-4352-95f8-5f31f60c7860" alt="image" width="300"/>

### Show the address of the 5 most expensive houses

| row          | address     | Price     | 
| :---         | :----  | :---- | 
| 1  | Zaferanieh  | 	9.240000e+10  | 
| 2  | Abazar  | 	9.100000e+10  | 
| 3  | Lavasan  | 	8.500000e+10  | 
| 4  | Ekhtiarieh  | 	8.160000e+10  | 
| 5  | Niavaran  | 		8.050000e+10  | 

### Compare your result with Scikit-Learn's results

Tehran House Price
| Models          | MAE    | MSE     | RMAE     |
| :---            | :----  | :---- | :---- |
| my LLS results  | 1188244052  | 2.00026e+18  | 1414306447  |
| sklearn results | 1191334095  | 2.00445e+18  | 1415789084  |
| RidgeCV results | 1193205134  | 2.00379e+18  | 1415554607  |

###  Why the MSE metric is a very large number?

Because Y numbers are very large and reach the power of two.

## Dollar Rial Price 💰

### Divide dataset to Ahmadinejad, Rouhani and Raisi's presidency

<img src="https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/c04c0583-5325-48d7-8c88-572799990c79" alt="image" width="500"/>

<img src="https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/96b609f4-4708-4fcc-b2f5-917361d9d2b8" alt="image" width="500"/>

<img src="https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/fdad4c33-5ed3-48c5-839f-6b79be979f66" alt="image" width="500"/>

### Show the highest dollar price in Ahmadinejad, Rouhani and Raisi's presidency respectively

| Presidency          | Ahmadinejad    | Rouhani     | Ebram     |
| :---            | :----  | :---- | :---- |
| min  | 13350  | 28880  | 253830  |

### Evaluate each model on test dataset using MAE loss function in Scikit-Learn

| Presidency          | Ahmadinejad    | Rouhani     | Ebram     |
| :---            | :----  | :---- | :---- |
| MAE  | 2821  | 32299  | 34187  |
