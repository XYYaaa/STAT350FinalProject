# STAT350FinalProject - Group 13
Group members: Hao Wu, Xingyu Yi, Mengying Yuan, Shuai Feng  
Professor: Derek Bingham  
For **full analysis and conclusion** please click here:
[final report pdf](https://github.com/hwa95/STAT350FinalProject/blob/main/result_paper.pdf).  
**Data** can be found on [this website](https://www.kaggle.com/mirichoi0218/insurance), or [csv file](https://github.com/hwa95/STAT350FinalProject/blob/main/insurance.csv).  
**R code** can be found [here.](https://github.com/hwa95/STAT350FinalProject/blob/main/code.Rmd)  
**Knitted html** of chunks and outputs please click here:
[code and output](https://github.com/hwa95/STAT350FinalProject/blob/main/code.html).  
## Abstract
In this article, we will study to determine the factors affecting medical costs in the United States. For the data set under study, 1338 data and 7 variables are collected, which are the response variable expenses and the 6 explanatory variables we want to study. After research for scatter plot matrix and complete linear regression model analysis, we used the AIC and BIC model selection steps and selected two models. Then we build comparative tests on these two selected models. Through the use of ANOVA, the final model was determined and showed the influence of medical expenses on the following factors: 7 explanatory variables, 11 interaction variables, of which 7 influencing factors are more important, namely SEX, CHILDREN, AGE square, the interaction term BMI & smoking, BMI & living in SOUTHEAST, smoking & BMI over 30, smoking, and age square.

## Methods
* Variable statistics
* AIC, BIC
* Forward, Backward, Stepwise
* Test for ANOVA table and plots

## Model
The seven important variables detected and included in the final model are SEX, CHILDREN, AGE2, the interaction term BMI: SMOKER, BMI: REGION (SOUTHEAST), SMOKER: BMI30, SMOKER: AGE2. 
Note: For the final model, please refer to the final report pdf.

## Conclusion
1. If the policy holder is male, the point estimation of charges will be 504.7322 lower than female, with other variables remaining constant.
2. The more children/dependents included in the insurance plan, the higher the cost, and the premium for one more child increases by 773.1290, while other variables remain unchanged.
3. The higher the square of the beneficiary’s age, the higher the premium. The square of the beneficiary’s age increases by 1 year and the cost increases by 4.4349, while other variables remain the same.
4. The BMI of smokers gains 1 unit, which is 504.6852 higher than the cost of non-smokers, while other variables remain unchanged.
5. The BMI of people living in SOUTHEAST increased by 1 unit, which is 129.0373 less than the cost of people living in other places, while other variables remain unchanged
6. People who smoke and have a BMI of more than 30 cost 14853.6719 more than others, while other variables remain the same.
7. The square of the age of the smoker increases by 1 unit, which is a decrease of 3.4516 compared to the cost of other people, while other variables remain unchanged.
Therefore, we know that the cost is related to the gender of the holder, the number of children/dependents, age, smoking, and BMI. Men need less expenses than women. The more children there are, the more expense is needed. Age is not linearly related to the cost. The older the age, the more the cost is. The cost of smokers is more than that of non-smokers, and the cost of smoking and overweight people is more than the cost of others.
## Recommend
1. Appropriate weight control can reduce corresponding expenses;
2. Reducing smoking can also reduce costs;
3. For people who smoke and are overweight, more control is needed.
 
