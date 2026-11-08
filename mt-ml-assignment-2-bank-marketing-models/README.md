# ML Assignment 2 — Bank Marketing Classification

## a. Problem statement

A Portuguese bank ran phone-based marketing campaigns to sell term deposits. Given client attributes and campaign contact history, the goal is to use the best fit model to predict whether a client will subscribe to a term deposit. Effectively a binary decision making problem with an answer in yes/no.

This project:

1. Trains five classification models on the same UCI Bank Marketing dataset
2. Evaluates each model with following analysis metrics:
   a. Accuracy
   b. AUC
   c. Precision
   d. Recall
   e. F1
   f. MCC
3. Serves the saved models through a Streamlit app that scores uploaded test CSVs without retraining. This is just a way to deploy the already worked models.

## b. Dataset description

- **Source:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Task:** Binary classification — predict term-deposit subscription (yes/no)
- **Instances:** 45,211; Data has class imbalance (88.3% no / 11.7% yes)
- **Input features:** There are 16 features — [age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome]
- **Preprocessing:** The following preprocessing has been applied to make data proper:
   a. one-hot encoding for categorical data
   b. standard scaling for numerical data
- **Split:** stratified 80/20 train/test; held-out set saved as `test_data.csv` (9,043 rows)

## c. Github Repository Link

- GitHub repository: https://github.com/wasiq-zia/applied-aiml
- Live Streamlit app: *(add Streamlit Community Cloud URL after deploy)*

## d. Models used

All five models were trained on the same train split and scored on the same `test_data.csv`.

### Comparison table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ---------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression    | 0.8457   | 0.9079 | 0.4182    | 0.8147 | 0.5527 | 0.5092 |
| Decision Tree          | 0.8301   | 0.8644 | 0.3934    | 0.8336 | 0.5345 | 0.4937 |
| KNN                    | 0.8962   | 0.8277 | 0.5990    | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes            | 0.8548   | 0.8101 | 0.4059    | 0.5198 | 0.4559 | 0.3774 |
| Support Vector Machine | 0.9010   | 0.9076 | 0.6463    | 0.3403 | 0.4458 | 0.4219 |

### Observations

| ML Model Name          | Observation about model performance |
| ---------------------- | ----------------------------------- |
| Logistic Regression    | Strong overall ranking model: highest AUC (0.9079), F1 (0.5527), and MCC (0.5092). It predicts most true subscribers (recall 0.8147) but also has more false positives (precision 0.4182). |
| Decision Tree          | Highest recall (0.8336) but lowest accuracy. Captures many “yes” cases while over-predicting positives; good MCC but weaker AUC than logistic/SVM. |
| KNN                    | High accuracy (0.8962) because of the class imbalance in favour of “no” class; recall is only 0.3403, so many subscribers with 'yes' are missed. Weaker ranking quality (AUC 0.8277). |
| Naive Bayes            | Average accuracy with the weakest AUC (0.8101) and MCC (0.3774) values. Feature independence assumptions are a poor fit. |
| Support Vector Machine | Best accuracy (0.9010) and precision (0.6463), with AUC nearly tied with logistic (0.9076). Recall is low (0.3403), so it detects lesser number of subscribers despite being high on accuracy. |
| Overall, Winner        | **Logistic Regression** — best balance for this imbalanced problem (top AUC, F1, and MCC) while still finding most true positives. We can prefer SVM to maximize accuracy/precision and if missed subscribers is acceptable for us. |

## How to run

```powershell
cd mt-ml-assignment-2-bank-marketing-models
pip install -r requirements.txt
python -m model.train       # retrain all five models (optional)
python -m model.evaluate    # reprint comparison metrics from saved models
streamlit run app.py        # upload test_data.csv and compare models
```

## Streamlit Community Cloud

1. Open [https://streamlit.io/cloud](https://streamlit.io/cloud) and connect this GitHub repo.
2. Set **Main file path** to `mt-ml-assignment-2-bank-marketing-models/app.py`.
3. Deploy, upload `test_data.csv`, and paste the live URL into section **c** above.
