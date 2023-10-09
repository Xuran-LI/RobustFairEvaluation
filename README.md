# RobustFair Evaluation 
Code for RobustFair: Adversarial Evaluation through Fairness Confusion Directed Gradient Search

# Package Requirements:
Python 3.8,

tensorflow 2.4.1,

numpy 1.19.5,

keras 2.4.3,

scikit-learn 1.0.2,

pandas 1.4.3,


# RobustFair for evaluating accurate fairness
This package provides code for evaluating accurate fairness using the RobustFair method. 
The following steps outline the process for conducting experiments on the Adult dataset:

1.Unzip the dataset.zip file.

2.Run adult_train_mode.py to train the baseline model.

3.Run adult_prepare_seeds.py to get the experimental seeds.

4.1.Run compare_experiment_RF.py to evaluate the accurate fairness.
4.2.Run check_loss_change_RobustFair.py to analyze the loss function trend during accurate fairness evaluation.

5.Run adult_retrain_model.py to retrain the model using the RobustFair evaluation from training dataset.

6.Run adult_check_BL_model.py and adult_check_retrain_model.py to check the models on the original testing data.

# Exporting experiment results

7.Use the adult_get_result.py script to export the experiment results as worksheets.


If you have any questions or need further assistance, please reach out to us.
