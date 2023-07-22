# K-Nearest Neighbors in ANSUR II dataset

![image](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/b9f91e36-21d7-4ca3-bccc-523767d7e567)


## Show heights for women and men on same plot.

![hist](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/be5e483c-1433-4ec5-be9d-c745425284ec)

### A. Why is the data of men higher than the data of women?

Because data of male is more than female (1985 women and 4081 men, respectively).

#‌## B. Why is the data of men more right than the data of women?

Because men are taller. Mean stature in male is 175 cm and in female is 163 cm.

For a better comparison, the following chart is suggested that has same density.

![hist density](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/b0054500-2d88-40c2-95fa-4aab952638e3)

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

## Plots

![output](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/079bc31b-b5a5-4763-9ad8-6a035ea14b20)

![Buttock Circumference](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/6c10e892-d031-4253-8814-8ae25727178c)

![fvsm](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/7506c9dc-b2e7-450c-a060-0ee816b87f15)


## Calculate confusion matrix for test dataset.

![Confusion Matrix](https://github.com/MSaberian/PyLearnMachineLearning/assets/43343453/f7d8693f-2d5b-4d11-9dda-01f4305076f6)

