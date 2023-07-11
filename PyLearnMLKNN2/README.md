# K-Nearest Neighbors in ANSUR II dataset

![image](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/b9f91e36-21d7-4ca3-bccc-523767d7e567)


## Show heights for women and men on same plot.

![hist](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/12b11784-8786-4787-b4c5-6e990452fe50)


### A. Why is the data of men higher than the data of women?

Because men are taller. Mean stature in male is 175 cm and in female is 163 cm.

#‌## B. Why is the data of men more right than the data of women?

Because men are heavier. Mean Weight in male is 86 kg and in female is 68 kg. 

## Evaluate your KNN algorithm on the test dataset with different values of k = 3, 5, 7, 10 and 15.

| k      | 3      | 5      | 7      | 10      | 15      |
| :---   | :----  | :----  | :----  | :----   | :----   |
| Score  | 83.1%  | 84.7%  | 84.9%  | 85.2%   | 85.1%   |

## Calculate confusion matrix for test dataset.

![Confusion Matrix me](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/611b3ead-8e54-4248-8a29-d27fa0f6b810)


## Evaluate the scikit-learn KNN algorithm on the test dataset. Make sure your accuracy is equal to scikit-learn's accuracy.

| k                                                                      | 3      | 5      | 7      | 10      | 15      |
| :---                                                                   | :----  | :----  | :----  | :----   | :----   |
| Scores obtained by weight and stature features                         | 83.3%  | 84.5%  | 84.8%  | 85.2%   | 85.1%   |
| Scores obtained by weight and Buttock Circumference features           | 97.9%  | 97.9%  | 98.1%  | 97.9%   | 97.9%   |
| Scores obtained by weight, stature and Buttock Circumference features  | 97.3%  | 97.3%  | 97.7%  | 97.5%   | 97.4%   |

## Calculate confusion matrix for test dataset.

![Confusion Matrix](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/f7d8693f-2d5b-4d11-9dda-01f4305076f6)

