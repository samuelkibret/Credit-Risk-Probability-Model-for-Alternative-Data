# Credit Risk Probability Model for Alternative Data

This project explores how to develop a redit risk probability model by cleaning, feature engineering, trainnig, evaluating, and hyperparametr-tuning.

## 📁 Project Structure
### Credit Risk Probability Model/
### ├── data/
### │  ├── raw/
### │  └── processed/
### ├── notebooks/
### ├── scripts/
### ├── src/
### ├── tests/
### ├── requirements.txt
### ├──.gitignore
### ├──docker-compose.yml
### ├──.github
### └── README.md


# Credit Scoring Business Understanding

## 1. The Basel II Accord focuses on measuring risks like credit or market risks to keep banks safe. This pushes banks to use models that are easy to understand and well-documented because:

#### - Regulators Need Clarity: Basel II wants banks to show how they measure risk. Simple models, like Logistic Regression, make it easy to show regulators how decisions are made.

#### - Capital Rules: Banks must hold enough money to cover risks. Clear models help calculate this accurately, and good documentation proves it’s done right.

#### - Better Risk Management: Understandable models help banks make smart decisions about loans or risks. Documentation keeps everything organized and consistent.

#### - Audits: Regulators and auditors check models. Clear models with good records are easier to review.

## 2. When we don’t have a clear "default" label (like knowing who didn’t pay their loan), we make a proxy variable using other data, like late payments or low credit scores.
### - Why We Need a Proxy:
#### * No Direct Data: Sometimes, we don’t have enough default records, especially for new loans or rare cases. A proxy helps us guess who might default.

#### * Rules and Business: Basel II asks banks to estimate default risks. A proxy lets us build models to do this and make decisions like who gets a loan.

#### * Model Training: Models need a target to predict. A proxy acts as that target when we don’t have real default data.

### - Risks of Using a Proxy:
#### * Wrong Predictions: The proxy might not match real defaults, leading to bad guesses. For example, using "late payments" might miss some risky borrowers “

## 3. In a regulated financial setting like Basel II, choosing between a simple, easy-to-understand model (like Logistic Regression with Weight of Evidence) and a complex, high-performing model (like Gradient Boosting) involves weighing pros and cons.

### - Simple Models (Logistic Regression with WoE):  

#### * Good: Easy to explain, good for regulators, stable, and cheap.  

#### * Bad: Less accurate, needs manual data work.

### - Complex Models (Gradient Boosting):  
#### * Good: Very accurate, handles messy data, good for business.  

#### * Bad: Hard to explain, might worry regulators, takes more work and money.

### Trade-offs: Simple models are better for rules and audits; complex models predict better but are harder to justify under Basel II. Banks might use both for different needs.





## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/samuelkibret/Credit-Risk-Probability-Model-for-Alternative-Data.git

cd Credit Risk Probability Model
```
### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows
pip install -r requirements.txt
```