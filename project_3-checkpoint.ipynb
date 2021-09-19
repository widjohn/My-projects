{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Data Science Project:\n",
    "using supervised machine learning algorithms in Python.\n",
    "\n",
    "\n",
    "Supervised Machine Learning is nothing but learning a function that maps an input to an output based on example input-output pairs. A supervised machine learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples.\n",
    "This is a supervised classification problem that we need to predict a binary outcome (Y/N).\n",
    "\n",
    "The project is broken into phases, representing each phase in the data science project lifecycle.\n",
    "Business Understand-Data Collection- EDA & Modelling the model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "***Business Understand.***\n",
    "The Thera Bank\n",
    "\n",
    "This case is about a bank (Thera Bank) whose management wants to explore ways of converting \n",
    "its liability customers to personal loan customers (while retaining them as depositors).\n",
    "A campaign that the bank ran last year for liability customers showed a healthy conversion\n",
    "rate of over 9% success. This has encouraged the retail marketing department to devise campaigns\n",
    "with better target marketing to increase the success ratio with minimal budget."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Goal is to Predict whether a customer will respond to a Personal Loan Campaign!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing libraries \n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from matplotlib import pyplot as plt\n",
    "import missingno as msno"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "metadata": {},
   "outputs": [],
   "source": [
    "#!pipenv install lightgbm --skip-lock"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing classes\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from xgboost import XGBClassifier, plot_importance\n",
    "#from lightgmb import LGMBClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.preprocessing import OneHotEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing libraries for Performance Measures\n",
    "from sklearn.metrics import confusion_matrix, classification_report,accuracy_score\n",
    "from sklearn.model_selection import StratifiedKFold,GridSearchCV\n",
    "from sklearn.base import clone\n",
    "from sklearn.model_selection import cross_val_score\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Data Collection**\n",
    "Data collection is the process of gathering and measuring information on variables of interest, in an established systematic fashion that enables one to answer stated research questions, test hypotheses, and evaluate outcomes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Loading the dataset\n",
    "loan= pd.read_csv('Bank_Personal_Loan_Modelling.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Age</th>\n",
       "      <th>Experience</th>\n",
       "      <th>Income</th>\n",
       "      <th>ZIP Code</th>\n",
       "      <th>Family</th>\n",
       "      <th>CCAvg</th>\n",
       "      <th>Education</th>\n",
       "      <th>Mortgage</th>\n",
       "      <th>Personal Loan</th>\n",
       "      <th>Securities Account</th>\n",
       "      <th>CD Account</th>\n",
       "      <th>Online</th>\n",
       "      <th>CreditCard</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>25</td>\n",
       "      <td>1</td>\n",
       "      <td>49</td>\n",
       "      <td>91107</td>\n",
       "      <td>4</td>\n",
       "      <td>1.6</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>45</td>\n",
       "      <td>19</td>\n",
       "      <td>34</td>\n",
       "      <td>90089</td>\n",
       "      <td>3</td>\n",
       "      <td>1.5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>39</td>\n",
       "      <td>15</td>\n",
       "      <td>11</td>\n",
       "      <td>94720</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>35</td>\n",
       "      <td>9</td>\n",
       "      <td>100</td>\n",
       "      <td>94112</td>\n",
       "      <td>1</td>\n",
       "      <td>2.7</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>35</td>\n",
       "      <td>8</td>\n",
       "      <td>45</td>\n",
       "      <td>91330</td>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID  Age  Experience  Income  ZIP Code  Family  CCAvg  Education  Mortgage  \\\n",
       "0   1   25           1      49     91107       4    1.6          1         0   \n",
       "1   2   45          19      34     90089       3    1.5          1         0   \n",
       "2   3   39          15      11     94720       1    1.0          1         0   \n",
       "3   4   35           9     100     94112       1    2.7          2         0   \n",
       "4   5   35           8      45     91330       4    1.0          2         0   \n",
       "\n",
       "   Personal Loan  Securities Account  CD Account  Online  CreditCard  \n",
       "0              0                   1           0       0           0  \n",
       "1              0                   1           0       0           0  \n",
       "2              0                   0           0       0           0  \n",
       "3              0                   0           0       0           0  \n",
       "4              0                   0           0       0           1  "
      ]
     },
     "execution_count": 186,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Let's look at the Top five rows in the dataset\n",
    "loan.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Making a copy of the original dataset\n",
    "loan1= loan.copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "EDA_\n",
    "Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ID                    0\n",
       "Age                   0\n",
       "Experience            0\n",
       "Income                0\n",
       "ZIP Code              0\n",
       "Family                0\n",
       "CCAvg                 0\n",
       "Education             0\n",
       "Mortgage              0\n",
       "Personal Loan         0\n",
       "Securities Account    0\n",
       "CD Account            0\n",
       "Online                0\n",
       "CreditCard            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 188,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Let's see if there are any missing value\n",
    "loan1.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we can see there are not any missing value in our dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 189,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check out any duplicated data\n",
    "loan1.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 190,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the max income of customers: 224\n"
     ]
    }
   ],
   "source": [
    "#Checking out the highest income of customers\n",
    "print(\"the max income of customers:\",loan1[\"Income\"].max())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1], dtype=int64)"
      ]
     },
     "execution_count": 191,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking out the creditcard column \n",
    "loan1[\"CreditCard\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.706\n",
       "1    0.294\n",
       "Name: CreditCard, dtype: float64"
      ]
     },
     "execution_count": 192,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#verifying how many customers have a credit card\n",
    "loan[\"CreditCard\"].value_counts(normalize= True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After verifying how many customers have a credit card,the credit card column has two values 1 and 0,1 means that the customer has a credit card ,and 0 means that the customer hasn't any credit card,with this insight I can confirm that only 29% of customers have a credit card."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "635"
      ]
     },
     "execution_count": 193,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Analyzing the column Mortgage\n",
    "loan[\"Mortgage\"].max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Age</th>\n",
       "      <th>Experience</th>\n",
       "      <th>Income</th>\n",
       "      <th>ZIP Code</th>\n",
       "      <th>Family</th>\n",
       "      <th>CCAvg</th>\n",
       "      <th>Education</th>\n",
       "      <th>Mortgage</th>\n",
       "      <th>Personal Loan</th>\n",
       "      <th>Securities Account</th>\n",
       "      <th>CD Account</th>\n",
       "      <th>Online</th>\n",
       "      <th>CreditCard</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2934</th>\n",
       "      <td>2935</td>\n",
       "      <td>37</td>\n",
       "      <td>13</td>\n",
       "      <td>195</td>\n",
       "      <td>91763</td>\n",
       "      <td>2</td>\n",
       "      <td>6.5</td>\n",
       "      <td>1</td>\n",
       "      <td>635</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        ID  Age  Experience  Income  ZIP Code  Family  CCAvg  Education  \\\n",
       "2934  2935   37          13     195     91763       2    6.5          1   \n",
       "\n",
       "      Mortgage  Personal Loan  Securities Account  CD Account  Online  \\\n",
       "2934       635              0                   0           0       1   \n",
       "\n",
       "      CreditCard  \n",
       "2934           0  "
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loan[loan[\"Mortgage\"]==635]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0], dtype=int64)"
      ]
     },
     "execution_count": 195,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loan[\"Securities Account\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.8956\n",
       "1    0.1044\n",
       "Name: Securities Account, dtype: float64"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loan[\"Securities Account\"].value_counts(normalize= True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After verifying how many customers have securities account,the securities account column has two values 1 and 0,1 means that the customer has securities account ,and 0 means that the customer hasn't any securities account,with this insight I can confirm that only 10.4% of customers have securities account."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Age</th>\n",
       "      <th>Experience</th>\n",
       "      <th>Income</th>\n",
       "      <th>ZIP Code</th>\n",
       "      <th>Family</th>\n",
       "      <th>CCAvg</th>\n",
       "      <th>Education</th>\n",
       "      <th>Mortgage</th>\n",
       "      <th>Personal Loan</th>\n",
       "      <th>Securities Account</th>\n",
       "      <th>CD Account</th>\n",
       "      <th>Online</th>\n",
       "      <th>CreditCard</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ID</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.008473</td>\n",
       "      <td>-0.008326</td>\n",
       "      <td>-0.017695</td>\n",
       "      <td>0.013432</td>\n",
       "      <td>-0.016797</td>\n",
       "      <td>-0.024675</td>\n",
       "      <td>0.021463</td>\n",
       "      <td>-0.013920</td>\n",
       "      <td>-0.024801</td>\n",
       "      <td>-0.016972</td>\n",
       "      <td>-0.006909</td>\n",
       "      <td>-0.002528</td>\n",
       "      <td>0.017028</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>-0.008473</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.994215</td>\n",
       "      <td>-0.055269</td>\n",
       "      <td>-0.029216</td>\n",
       "      <td>-0.046418</td>\n",
       "      <td>-0.052012</td>\n",
       "      <td>0.041334</td>\n",
       "      <td>-0.012539</td>\n",
       "      <td>-0.007726</td>\n",
       "      <td>-0.000436</td>\n",
       "      <td>0.008043</td>\n",
       "      <td>0.013702</td>\n",
       "      <td>0.007681</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Experience</th>\n",
       "      <td>-0.008326</td>\n",
       "      <td>0.994215</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.046574</td>\n",
       "      <td>-0.028626</td>\n",
       "      <td>-0.052563</td>\n",
       "      <td>-0.050077</td>\n",
       "      <td>0.013152</td>\n",
       "      <td>-0.010582</td>\n",
       "      <td>-0.007413</td>\n",
       "      <td>-0.001232</td>\n",
       "      <td>0.010353</td>\n",
       "      <td>0.013898</td>\n",
       "      <td>0.008967</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Income</th>\n",
       "      <td>-0.017695</td>\n",
       "      <td>-0.055269</td>\n",
       "      <td>-0.046574</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.016410</td>\n",
       "      <td>-0.157501</td>\n",
       "      <td>0.645984</td>\n",
       "      <td>-0.187524</td>\n",
       "      <td>0.206806</td>\n",
       "      <td>0.502462</td>\n",
       "      <td>-0.002616</td>\n",
       "      <td>0.169738</td>\n",
       "      <td>0.014206</td>\n",
       "      <td>-0.002385</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ZIP Code</th>\n",
       "      <td>0.013432</td>\n",
       "      <td>-0.029216</td>\n",
       "      <td>-0.028626</td>\n",
       "      <td>-0.016410</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.011778</td>\n",
       "      <td>-0.004061</td>\n",
       "      <td>-0.017377</td>\n",
       "      <td>0.007383</td>\n",
       "      <td>0.000107</td>\n",
       "      <td>0.004704</td>\n",
       "      <td>0.019972</td>\n",
       "      <td>0.016990</td>\n",
       "      <td>0.007691</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Family</th>\n",
       "      <td>-0.016797</td>\n",
       "      <td>-0.046418</td>\n",
       "      <td>-0.052563</td>\n",
       "      <td>-0.157501</td>\n",
       "      <td>0.011778</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.109275</td>\n",
       "      <td>0.064929</td>\n",
       "      <td>-0.020445</td>\n",
       "      <td>0.061367</td>\n",
       "      <td>0.019994</td>\n",
       "      <td>0.014110</td>\n",
       "      <td>0.010354</td>\n",
       "      <td>0.011588</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>CCAvg</th>\n",
       "      <td>-0.024675</td>\n",
       "      <td>-0.052012</td>\n",
       "      <td>-0.050077</td>\n",
       "      <td>0.645984</td>\n",
       "      <td>-0.004061</td>\n",
       "      <td>-0.109275</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.136124</td>\n",
       "      <td>0.109905</td>\n",
       "      <td>0.366889</td>\n",
       "      <td>0.015086</td>\n",
       "      <td>0.136534</td>\n",
       "      <td>-0.003611</td>\n",
       "      <td>-0.006689</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Education</th>\n",
       "      <td>0.021463</td>\n",
       "      <td>0.041334</td>\n",
       "      <td>0.013152</td>\n",
       "      <td>-0.187524</td>\n",
       "      <td>-0.017377</td>\n",
       "      <td>0.064929</td>\n",
       "      <td>-0.136124</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.033327</td>\n",
       "      <td>0.136722</td>\n",
       "      <td>-0.010812</td>\n",
       "      <td>0.013934</td>\n",
       "      <td>-0.015004</td>\n",
       "      <td>-0.011014</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mortgage</th>\n",
       "      <td>-0.013920</td>\n",
       "      <td>-0.012539</td>\n",
       "      <td>-0.010582</td>\n",
       "      <td>0.206806</td>\n",
       "      <td>0.007383</td>\n",
       "      <td>-0.020445</td>\n",
       "      <td>0.109905</td>\n",
       "      <td>-0.033327</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.142095</td>\n",
       "      <td>-0.005411</td>\n",
       "      <td>0.089311</td>\n",
       "      <td>-0.005995</td>\n",
       "      <td>-0.007231</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Personal Loan</th>\n",
       "      <td>-0.024801</td>\n",
       "      <td>-0.007726</td>\n",
       "      <td>-0.007413</td>\n",
       "      <td>0.502462</td>\n",
       "      <td>0.000107</td>\n",
       "      <td>0.061367</td>\n",
       "      <td>0.366889</td>\n",
       "      <td>0.136722</td>\n",
       "      <td>0.142095</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.021954</td>\n",
       "      <td>0.316355</td>\n",
       "      <td>0.006278</td>\n",
       "      <td>0.002802</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Securities Account</th>\n",
       "      <td>-0.016972</td>\n",
       "      <td>-0.000436</td>\n",
       "      <td>-0.001232</td>\n",
       "      <td>-0.002616</td>\n",
       "      <td>0.004704</td>\n",
       "      <td>0.019994</td>\n",
       "      <td>0.015086</td>\n",
       "      <td>-0.010812</td>\n",
       "      <td>-0.005411</td>\n",
       "      <td>0.021954</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.317034</td>\n",
       "      <td>0.012627</td>\n",
       "      <td>-0.015028</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>CD Account</th>\n",
       "      <td>-0.006909</td>\n",
       "      <td>0.008043</td>\n",
       "      <td>0.010353</td>\n",
       "      <td>0.169738</td>\n",
       "      <td>0.019972</td>\n",
       "      <td>0.014110</td>\n",
       "      <td>0.136534</td>\n",
       "      <td>0.013934</td>\n",
       "      <td>0.089311</td>\n",
       "      <td>0.316355</td>\n",
       "      <td>0.317034</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.175880</td>\n",
       "      <td>0.278644</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Online</th>\n",
       "      <td>-0.002528</td>\n",
       "      <td>0.013702</td>\n",
       "      <td>0.013898</td>\n",
       "      <td>0.014206</td>\n",
       "      <td>0.016990</td>\n",
       "      <td>0.010354</td>\n",
       "      <td>-0.003611</td>\n",
       "      <td>-0.015004</td>\n",
       "      <td>-0.005995</td>\n",
       "      <td>0.006278</td>\n",
       "      <td>0.012627</td>\n",
       "      <td>0.175880</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.004210</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>CreditCard</th>\n",
       "      <td>0.017028</td>\n",
       "      <td>0.007681</td>\n",
       "      <td>0.008967</td>\n",
       "      <td>-0.002385</td>\n",
       "      <td>0.007691</td>\n",
       "      <td>0.011588</td>\n",
       "      <td>-0.006689</td>\n",
       "      <td>-0.011014</td>\n",
       "      <td>-0.007231</td>\n",
       "      <td>0.002802</td>\n",
       "      <td>-0.015028</td>\n",
       "      <td>0.278644</td>\n",
       "      <td>0.004210</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          ID       Age  Experience    Income  ZIP Code  \\\n",
       "ID                  1.000000 -0.008473   -0.008326 -0.017695  0.013432   \n",
       "Age                -0.008473  1.000000    0.994215 -0.055269 -0.029216   \n",
       "Experience         -0.008326  0.994215    1.000000 -0.046574 -0.028626   \n",
       "Income             -0.017695 -0.055269   -0.046574  1.000000 -0.016410   \n",
       "ZIP Code            0.013432 -0.029216   -0.028626 -0.016410  1.000000   \n",
       "Family             -0.016797 -0.046418   -0.052563 -0.157501  0.011778   \n",
       "CCAvg              -0.024675 -0.052012   -0.050077  0.645984 -0.004061   \n",
       "Education           0.021463  0.041334    0.013152 -0.187524 -0.017377   \n",
       "Mortgage           -0.013920 -0.012539   -0.010582  0.206806  0.007383   \n",
       "Personal Loan      -0.024801 -0.007726   -0.007413  0.502462  0.000107   \n",
       "Securities Account -0.016972 -0.000436   -0.001232 -0.002616  0.004704   \n",
       "CD Account         -0.006909  0.008043    0.010353  0.169738  0.019972   \n",
       "Online             -0.002528  0.013702    0.013898  0.014206  0.016990   \n",
       "CreditCard          0.017028  0.007681    0.008967 -0.002385  0.007691   \n",
       "\n",
       "                      Family     CCAvg  Education  Mortgage  Personal Loan  \\\n",
       "ID                 -0.016797 -0.024675   0.021463 -0.013920      -0.024801   \n",
       "Age                -0.046418 -0.052012   0.041334 -0.012539      -0.007726   \n",
       "Experience         -0.052563 -0.050077   0.013152 -0.010582      -0.007413   \n",
       "Income             -0.157501  0.645984  -0.187524  0.206806       0.502462   \n",
       "ZIP Code            0.011778 -0.004061  -0.017377  0.007383       0.000107   \n",
       "Family              1.000000 -0.109275   0.064929 -0.020445       0.061367   \n",
       "CCAvg              -0.109275  1.000000  -0.136124  0.109905       0.366889   \n",
       "Education           0.064929 -0.136124   1.000000 -0.033327       0.136722   \n",
       "Mortgage           -0.020445  0.109905  -0.033327  1.000000       0.142095   \n",
       "Personal Loan       0.061367  0.366889   0.136722  0.142095       1.000000   \n",
       "Securities Account  0.019994  0.015086  -0.010812 -0.005411       0.021954   \n",
       "CD Account          0.014110  0.136534   0.013934  0.089311       0.316355   \n",
       "Online              0.010354 -0.003611  -0.015004 -0.005995       0.006278   \n",
       "CreditCard          0.011588 -0.006689  -0.011014 -0.007231       0.002802   \n",
       "\n",
       "                    Securities Account  CD Account    Online  CreditCard  \n",
       "ID                           -0.016972   -0.006909 -0.002528    0.017028  \n",
       "Age                          -0.000436    0.008043  0.013702    0.007681  \n",
       "Experience                   -0.001232    0.010353  0.013898    0.008967  \n",
       "Income                       -0.002616    0.169738  0.014206   -0.002385  \n",
       "ZIP Code                      0.004704    0.019972  0.016990    0.007691  \n",
       "Family                        0.019994    0.014110  0.010354    0.011588  \n",
       "CCAvg                         0.015086    0.136534 -0.003611   -0.006689  \n",
       "Education                    -0.010812    0.013934 -0.015004   -0.011014  \n",
       "Mortgage                     -0.005411    0.089311 -0.005995   -0.007231  \n",
       "Personal Loan                 0.021954    0.316355  0.006278    0.002802  \n",
       "Securities Account            1.000000    0.317034  0.012627   -0.015028  \n",
       "CD Account                    0.317034    1.000000  0.175880    0.278644  \n",
       "Online                        0.012627    0.175880  1.000000    0.004210  \n",
       "CreditCard                   -0.015028    0.278644  0.004210    1.000000  "
      ]
     },
     "execution_count": 197,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#check out the correlation between the attributes\n",
    "correlation= loan1.corr()\n",
    "correlation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Age</th>\n",
       "      <th>Experience</th>\n",
       "      <th>Income</th>\n",
       "      <th>ZIP Code</th>\n",
       "      <th>Family</th>\n",
       "      <th>CCAvg</th>\n",
       "      <th>Education</th>\n",
       "      <th>Mortgage</th>\n",
       "      <th>Personal Loan</th>\n",
       "      <th>Securities Account</th>\n",
       "      <th>CD Account</th>\n",
       "      <th>Online</th>\n",
       "      <th>CreditCard</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.00000</td>\n",
       "      <td>5000.000000</td>\n",
       "      <td>5000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2500.500000</td>\n",
       "      <td>45.338400</td>\n",
       "      <td>20.104600</td>\n",
       "      <td>73.774200</td>\n",
       "      <td>93152.503000</td>\n",
       "      <td>2.396400</td>\n",
       "      <td>1.937938</td>\n",
       "      <td>1.881000</td>\n",
       "      <td>56.498800</td>\n",
       "      <td>0.096000</td>\n",
       "      <td>0.104400</td>\n",
       "      <td>0.06040</td>\n",
       "      <td>0.596800</td>\n",
       "      <td>0.294000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1443.520003</td>\n",
       "      <td>11.463166</td>\n",
       "      <td>11.467954</td>\n",
       "      <td>46.033729</td>\n",
       "      <td>2121.852197</td>\n",
       "      <td>1.147663</td>\n",
       "      <td>1.747659</td>\n",
       "      <td>0.839869</td>\n",
       "      <td>101.713802</td>\n",
       "      <td>0.294621</td>\n",
       "      <td>0.305809</td>\n",
       "      <td>0.23825</td>\n",
       "      <td>0.490589</td>\n",
       "      <td>0.455637</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>23.000000</td>\n",
       "      <td>-3.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>9307.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1250.750000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>39.000000</td>\n",
       "      <td>91911.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.700000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>2500.500000</td>\n",
       "      <td>45.000000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>64.000000</td>\n",
       "      <td>93437.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.500000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>3750.250000</td>\n",
       "      <td>55.000000</td>\n",
       "      <td>30.000000</td>\n",
       "      <td>98.000000</td>\n",
       "      <td>94608.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>2.500000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>101.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>5000.000000</td>\n",
       "      <td>67.000000</td>\n",
       "      <td>43.000000</td>\n",
       "      <td>224.000000</td>\n",
       "      <td>96651.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>635.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                ID          Age   Experience       Income      ZIP Code  \\\n",
       "count  5000.000000  5000.000000  5000.000000  5000.000000   5000.000000   \n",
       "mean   2500.500000    45.338400    20.104600    73.774200  93152.503000   \n",
       "std    1443.520003    11.463166    11.467954    46.033729   2121.852197   \n",
       "min       1.000000    23.000000    -3.000000     8.000000   9307.000000   \n",
       "25%    1250.750000    35.000000    10.000000    39.000000  91911.000000   \n",
       "50%    2500.500000    45.000000    20.000000    64.000000  93437.000000   \n",
       "75%    3750.250000    55.000000    30.000000    98.000000  94608.000000   \n",
       "max    5000.000000    67.000000    43.000000   224.000000  96651.000000   \n",
       "\n",
       "            Family        CCAvg    Education     Mortgage  Personal Loan  \\\n",
       "count  5000.000000  5000.000000  5000.000000  5000.000000    5000.000000   \n",
       "mean      2.396400     1.937938     1.881000    56.498800       0.096000   \n",
       "std       1.147663     1.747659     0.839869   101.713802       0.294621   \n",
       "min       1.000000     0.000000     1.000000     0.000000       0.000000   \n",
       "25%       1.000000     0.700000     1.000000     0.000000       0.000000   \n",
       "50%       2.000000     1.500000     2.000000     0.000000       0.000000   \n",
       "75%       3.000000     2.500000     3.000000   101.000000       0.000000   \n",
       "max       4.000000    10.000000     3.000000   635.000000       1.000000   \n",
       "\n",
       "       Securities Account  CD Account       Online   CreditCard  \n",
       "count         5000.000000  5000.00000  5000.000000  5000.000000  \n",
       "mean             0.104400     0.06040     0.596800     0.294000  \n",
       "std              0.305809     0.23825     0.490589     0.455637  \n",
       "min              0.000000     0.00000     0.000000     0.000000  \n",
       "25%              0.000000     0.00000     0.000000     0.000000  \n",
       "50%              0.000000     0.00000     1.000000     0.000000  \n",
       "75%              0.000000     0.00000     1.000000     1.000000  \n",
       "max              1.000000     1.00000     1.000000     1.000000  "
      ]
     },
     "execution_count": 198,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Let's summarizing statistics using pandas describe() method\n",
    "loan1.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3], dtype=int64)"
      ]
     },
     "execution_count": 200,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#verifying the education of the customers\n",
    "loan1[\"Education\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    0.4192\n",
       "3    0.3002\n",
       "2    0.2806\n",
       "Name: Education, dtype: float64"
      ]
     },
     "execution_count": 201,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Analyzing customer's education\n",
    "loan1[\"Education\"].value_counts(normalize= True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here as we can notice the education column contains three values ,1 ,2 and 3,maybe 1 means customer has a bachelor's degree, 2 customer has a master's degree and 3 customer has doctoral degree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.9396\n",
       "1    0.0604\n",
       "Name: CD Account, dtype: float64"
      ]
     },
     "execution_count": 202,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Analysing the cd account of the customers\n",
    "loan1[\"CD Account\"].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After analyzing the CD Account colmun,i can confirm 93% of customers don't have any CD Account in the bank."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 203,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the max values of experience of customers: 43\n"
     ]
    }
   ],
   "source": [
    "print(\"the max values of experience of customers:\",loan1[\"Experience\"].max())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ID                   -0.024801\n",
       "Age                  -0.007726\n",
       "Experience           -0.007413\n",
       "ZIP Code              0.000107\n",
       "CreditCard            0.002802\n",
       "Online                0.006278\n",
       "Securities Account    0.021954\n",
       "Family                0.061367\n",
       "Education             0.136722\n",
       "Mortgage              0.142095\n",
       "CD Account            0.316355\n",
       "CCAvg                 0.366889\n",
       "Income                0.502462\n",
       "Personal Loan         1.000000\n",
       "Name: Personal Loan, dtype: float64"
      ]
     },
     "execution_count": 205,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Verifying how look like the correlation between the target variable and independent variables\n",
    "corre_target= loan1.corr()['Personal Loan'].sort_values()\n",
    "corre_target"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After checking out the correlation between  the dependent variable and independent variables, only the Income has a correlation with \n",
    "the dependent variable"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Check target variable distribution: \n",
    "Let’s look at the distribution of personal loan values. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.904\n",
       "1    0.096\n",
       "Name: Personal Loan, dtype: float64"
      ]
     },
     "execution_count": 206,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#let's see how many values total 0 and 1 in the target\n",
    "\n",
    "loan1['Personal Loan'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here as we can see,in the last campaign only 9.6% of customers accepted the personal loan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Education            -0.187524\n",
       "Family               -0.157501\n",
       "Age                  -0.055269\n",
       "Experience           -0.046574\n",
       "ID                   -0.017695\n",
       "ZIP Code             -0.016410\n",
       "Securities Account   -0.002616\n",
       "CreditCard           -0.002385\n",
       "Online                0.014206\n",
       "CD Account            0.169738\n",
       "Mortgage              0.206806\n",
       "Personal Loan         0.502462\n",
       "CCAvg                 0.645984\n",
       "Income                1.000000\n",
       "Name: Income, dtype: float64"
      ]
     },
     "execution_count": 207,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corre_income= loan1.corr()['Income'].sort_values()\n",
    "corre_income"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Pearson Correlation: Measure the strength of the correlation between two features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "metadata": {},
   "outputs": [],
   "source": [
    "#let's apply the pearson correlation\n",
    "from scipy import stats\n",
    "pearson_coef, p_value=stats.pearsonr(loan1[\"Personal Loan\"],loan1[\"Income\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the pearson correlation: 0.5024622924949431\n"
     ]
    }
   ],
   "source": [
    "print(\"the pearson correlation:\", pearson_coef)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the P_value: 3.560286e-318\n"
     ]
    }
   ],
   "source": [
    "print(\"the P_value:\", p_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "metadata": {},
   "outputs": [],
   "source": [
    "#let's apply the pearson correlation\n",
    "from scipy import stats\n",
    "pearson_coef, p_value=stats.pearsonr(loan1[\"Education\"],loan1[\"Age\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.04133438336063412"
      ]
     },
     "execution_count": 212,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pearson_coef"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Discover and Visualize the Data to Gain Insights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x24ff5937850>"
      ]
     },
     "execution_count": 213,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#studying the data distribution in each attribute\n",
    "sns.distplot(loan1['Income'], bins= 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "valor skewness= 0.8410861846424931\n"
     ]
    }
   ],
   "source": [
    "#let's check Skewness\n",
    "from scipy import stats\n",
    "skewness= stats.skew(loan1.Income)\n",
    "print('valor skewness=', skewness)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Note:After verifying the distribution,we see that the distribution of skewness equal to 0.841, that means we have a positive skewness"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x24ff72f5430>"
      ]
     },
     "execution_count": 215,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1800x720 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Using missingno to identify missing data\n",
    "msno.bar(loan1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here as we can see,the barplot provides a simple plot where each bar represents a column within the DataFrame,it shows us there aren't any missing value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x24ff73f8b20>"
      ]
     },
     "execution_count": 216,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#studying the data distribution in each attribute\n",
    "sns.boxplot(x=\"Income\", data=loan1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As we can see there are some outliers in the Income feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x24ff7446250>"
      ]
     },
     "execution_count": 217,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Analyzing personal loan and credit card columns with a countplot\n",
    "sns.countplot(x='Personal Loan', hue='CreditCard', data=loan1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we can extract a few insights from this plot, most of customers who didn't accept the personal loan,they don't have a credit card either."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 218,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.706\n",
       "1    0.294\n",
       "Name: CreditCard, dtype: float64"
      ]
     },
     "execution_count": 218,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loan1[\"CreditCard\"].value_counts(normalize= True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 219,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.904\n",
       "1    0.096\n",
       "Name: Personal Loan, dtype: float64"
      ]
     },
     "execution_count": 219,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loan1[\"Personal Loan\"].value_counts(normalize= True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 220,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x24ff74a2fa0>"
      ]
     },
     "execution_count": 220,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYsAAAEGCAYAAACUzrmNAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAWFklEQVR4nO3dfZBV9Z3n8fdXRFEjbhB0lcaAxtoIKhgQnajxgdkSNSPqGAsSFR8iq2XEVDaZ0qRqxswUVZlKraUkE6uMo4JrwZDM4FP5uAo6PkSFxYmAMZpAtNUAYjZqGAXa7/7RB7yBpn8X07dvw32/qm71ud97fud8bxf66d85554bmYkkSd3ZpdkNSJL6PsNCklRkWEiSigwLSVKRYSFJKtq12Q00yuDBg3P48OHNbkOSdiiLFy9+OzOHbFnfacNi+PDhLFq0qNltSNIOJSJ+21Xdw1CSpCLDQpJUZFhIkop22nMWktSVDRs20N7ezgcffNDsVppqwIABtLW10b9//7rWNywktZT29nb23ntvhg8fTkQ0u52myEzWrl1Le3s7I0aMqGuMh6EktZQPPviAfffdt2WDAiAi2HfffbdrdmVYSGo5rRwUm2zv78CwkCQVGRaSWt7vfvc7Jk+ezCGHHMLIkSM5/fTT+dWvfsXKlSvZY489OOqoozjssMMYP348s2bN6nZbV199NUOHDuWjjz7qpe637YYbbmDdunU9si1PcG/D2G/PbnYLfcbiH1zY7BakhslMzj77bKZOncrcuXMBeOGFF1i1ahXDhg3jkEMOYcmSJQD85je/4ZxzzuGjjz7i4osv3mpbH330EfPnz2fYsGE88cQTnHTSSb35VrZyww03cP7557Pnnnv+2dtyZiGppS1YsID+/ftz+eWXb66NGTOGE044Yat1Dz74YK6//npmzpy5zW0dfvjhXHHFFcyZM2dzfdWqVZx99tmMHj2a0aNH8/TTTwMwe/ZsjjzySEaPHs0FF1wAwG9/+1smTJjAkUceyYQJE3jttdcAuOiii/jZz362eZuf+tSnAFi4cCEnnXQS5557Lp/73Of46le/SmYyc+ZM3nzzTU4++WROPvnkP/O35MxCUotbunQpY8eOrXv9z3/+8/zyl7/s8rU5c+YwZcoUJk2axHe+8x02bNhA//79mT59OieeeCLz58+no6OD999/n2XLljFjxgyeeuopBg8ezDvvvAPA17/+dS688EKmTp3KrbfeyvTp07nrrru67WnJkiUsW7aMAw88kOOOO46nnnqK6dOnc/3117NgwQIGDx5c/y9kG5xZSNJ2yMwu6+vXr+f+++/nrLPOYuDAgRxzzDE8/PDDADz22GNcccUVAPTr14999tmHxx57jHPPPXfz/8gHDRoEwDPPPMNXvvIVAC644AKefPLJYk/jx4+nra2NXXbZhTFjxrBy5co/921uxZmFpJY2atSoPzm8U7JkyRIOO+ywreoPPvggf/jDHzjiiCMAWLduHXvuuSdnnHFGl9vJzLouX920zq677rr5pHlmsn79+s3r7L777puX+/Xrx8aNG+t+P/VyZiGppZ1yyil8+OGH/OQnP9lce/7553n88ce3WnflypV861vf4qqrrtrqtTlz5nDLLbewcuVKVq5cyYoVK3j44YdZt24dEyZM4KabbgKgo6ODd999lwkTJjBv3jzWrl0LsPkw1Be+8IXNJ9rvvPNOjj/+eKDzaxcWL14MwN13382GDRuK723vvffmvffe255fxzYZFpJaWkQwf/58HnnkEQ455BBGjRrFddddx4EHHgjAr3/9682Xzp533nlcddVVW10JtW7dOh566KE/mUXstddeHH/88dx7773ceOONLFiwgCOOOIKxY8eybNkyRo0axXe/+11OPPFERo8ezTe/+U0AZs6cyW233caRRx7JHXfcwY033gjAZZddxuOPP8748eN59tln2WuvvYrvbdq0aZx22mk9coI7tnX8bUc3bty4/HO+/MhLZz/mpbPambz00ktdHkZqRV39LiJicWaO23JdZxaSpCLDQpJUZFhIkooMC0lSkWEhSSoyLCRJRQ3/BHdE9AMWAW9k5pciYhDwL8BwYCVwXmb+vlr3WuBSoAOYnpkPVfWxwO3AHsD9wNW5s17zK6lP6+nL6uu9NP3BBx/k6quvpqOjg6997Wtcc801PdpHSW/MLK4GXqp5fg3waGYeCjxaPSciRgKTgVHARODHVdAA3ARMAw6tHhN7oW9J6hM6Ojq48soreeCBB1i+fDlz5sxh+fLlvdpDQ8MiItqAM4BbasqTgE3fHjILOKumPjczP8zMFcCrwPiIOAAYmJnPVLOJ2TVjJGmn99xzz/HZz36Wgw8+mN12243Jkydz9913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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Analyzing personal loan and credit card columns with a countplot\n",
    "sns.countplot(x='Personal Loan', hue='CD Account', data=loan1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Here when we are analyzing personal loan and CD Account together we can confirm, \n",
    "most of customers who didn't accept the personal loan,they don't have a CD Account  either."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 221,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x24ff74f8d30>"
      ]
     },
     "execution_count": 221,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Seeing the target's distribution\n",
    "sns.distplot(loan1['CCAvg'], bins= 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "valor skewness= 1.5979637637001873\n"
     ]
    }
   ],
   "source": [
    "#let's check Skewness\n",
    "from scipy import stats\n",
    "skewness= stats.skew(loan1.CCAvg)\n",
    "print('valor skewness=', skewness)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Note:After verifying the distribution,we see that the distribution of skewness equal to 1.597 that means we have a positive skewness"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Exploratory Data Analysis Concluding  Remarks:\n",
    "\n",
    "*The dataset doesn't have any missing value\n",
    "*Most of customer don't have a credit card in the bank\n",
    "*Most of customer don't have any CD Account in the bank\n",
    "*In last campaign only 9.6% accepted the personal loan\n",
    "*Most of customer who did not accepte the personal loan ,they have neither credit card nor CD Account in the Bank\n",
    "*In the dataset ,the Income column has a correlation with the dependent variable \n",
    "*The Education column has a correlation with the age column\n",
    "*we have some positive skewness in the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5000, 11)"
      ]
     },
     "execution_count": 223,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Let's create the input feature X and our target y\n",
    "x= loan1.drop(['ID','ZIP Code','Personal Loan'], axis= 1).values\n",
    "x.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 224,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5000,)"
      ]
     },
     "execution_count": 224,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y= loan1['Personal Loan'].values\n",
    "y.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Divide the Dataset into  training and  testing set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((3500, 11), (1500, 11))"
      ]
     },
     "execution_count": 225,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#let's split the data into training set and testing set\n",
    "X_train, X_test, Y_train, Y_test= train_test_split(x, y, test_size= 0.3, random_state= 40)\n",
    "X_train.shape, X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 226,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the training set valor of the X_train is:(3500, 11)\n",
      "the testing set valor of the X_test is :(1500, 11)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "((3500,), (1500,))"
      ]
     },
     "execution_count": 226,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(\"the training set valor of the X_train is:{}\".format(X_train.shape))\n",
    "print(\"the testing set valor of the X_test is :{}\".format(X_test.shape))\n",
    "\n",
    "Y_train.shape, Y_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Feature Scaling:\n",
    "One of the most important transformations you need to apply to your data is feature\n",
    "scaling. With few exceptions, Machine Learning algorithms don’t perform well when\n",
    "the input numerical attributes have very different scales. This is the case for the Bank personal Loan data:\n",
    "Feature scaling to make sure that all features are on a similar scale,more generally when we are performing feature scaling ,what we often want to do is get every feature into aproximately -1 to +1 range."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 227,
   "metadata": {},
   "outputs": [],
   "source": [
    "standard= StandardScaler()\n",
    "X_train= standard.fit_transform(X_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 228,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.2777534 ,  1.21145449, -0.88602563, ..., -0.24683507,\n",
       "         0.8223413 , -0.64620273],\n",
       "       [ 1.36491087,  1.47268859, -1.10113559, ..., -0.24683507,\n",
       "        -1.21604011, -0.64620273],\n",
       "       [ 1.10343846,  1.03729842, -0.71393766, ..., -0.24683507,\n",
       "         0.8223413 , -0.64620273],\n",
       "       ...,\n",
       "       [-1.42412815, -1.48796457, -0.43429472, ..., -0.24683507,\n",
       "        -1.21604011, -0.64620273],\n",
       "       [ 1.19059593,  1.12437645,  0.18952417, ..., -0.24683507,\n",
       "         0.8223413 , -0.64620273],\n",
       "       [-0.46539598, -0.53010619, -0.26220675, ..., -0.24683507,\n",
       "         0.8223413 ,  1.54750198]])"
      ]
     },
     "execution_count": 228,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 229,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([11.47348595, 11.48395243, 46.4878522 ,  1.14590722,  1.77822143,\n",
       "        0.84207106, 99.04455451,  0.30189537,  0.23265969,  0.49058532,\n",
       "        0.45584987])"
      ]
     },
     "execution_count": 229,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "standard.scale_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 230,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test=standard.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Select and Train a Model:\n",
    "Training and Evaluating on the Training Set:\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Model Training:\n",
    "Naive Bayes classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 231,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GaussianNB()"
      ]
     },
     "execution_count": 231,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Fitting the Naive Bayes classifier \n",
    "naive= GaussianNB()\n",
    "naive.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Testing Model on the testing set*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 232,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 1, 0], dtype=int64)"
      ]
     },
     "execution_count": 232,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred_xtest= naive.predict(X_test)\n",
    "pred_xtest"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Check out the Performance of our classificatiom model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 233,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.96      0.91      0.93      1363\n",
      "           1       0.39      0.58      0.46       137\n",
      "\n",
      "    accuracy                           0.88      1500\n",
      "   macro avg       0.67      0.74      0.70      1500\n",
      "weighted avg       0.90      0.88      0.89      1500\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(Y_test, pred_xtest))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 234,
   "metadata": {},
   "outputs": [],
   "source": [
    "#performance measure\n",
    "Accuracy= accuracy_score(Y_test, pred_xtest)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 235,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " accuracy: 0.8786666666666667\n"
     ]
    }
   ],
   "source": [
    "print(' accuracy:', Accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 236,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.90428571, 0.88857143, 0.87714286, 0.88142857, 0.86857143])"
      ]
     },
     "execution_count": 236,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "naive_cross= cross_val_score(naive, X_train,Y_train, cv=5,scoring=\"accuracy\")\n",
    "naive_cross"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 237,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SGDClassifier(random_state=42)"
      ]
     },
     "execution_count": 237,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Fitting the Stochastic Gradient Descent(SGD) classifier \n",
    "from sklearn.linear_model import SGDClassifier\n",
    "sgd_clf = SGDClassifier(random_state=42)\n",
    "sgd_clf.fit(X_train,Y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 238,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 238,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#testing \n",
    "sgd_pred= sgd_clf.predict(X_test)\n",
    "sgd_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 239,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the accuracy of sgdclassifier: 0.9546666666666667\n"
     ]
    }
   ],
   "source": [
    "#performance measure\n",
    "sgd_accuracy= accuracy_score(Y_test, sgd_pred)\n",
    "print(\"the accuracy of sgdclassifier:\",sgd_accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Measuring Accuracy Using Cross-Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 240,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the acccuracy of sgdclassifier using cross validation [0.94285714 0.94285714 0.93857143 0.95142857 0.93285714]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "sgd_cross= cross_val_score(sgd_clf, X_train,Y_train, cv=5, scoring=\"accuracy\")\n",
    "print(\"the acccuracy of sgdclassifier using cross validation\",sgd_cross)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Random Forest Classifier Class*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(n_estimators=200)"
      ]
     },
     "execution_count": 141,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Fitting the Random Forest Classifier \n",
    "rfc= RandomForestClassifier(n_estimators= 200)\n",
    "rfc.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Testing model on the testing set*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rfc_xtest= rfc.predict(X_test)\n",
    "rfc_xtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the accuracy of random forest classifier: 0.9873333333333333\n"
     ]
    }
   ],
   "source": [
    "#calculating the accuracy of the model with X_test\n",
    "accuracy_xtest= accuracy_score(Y_test, rfc_xtest)\n",
    "print('the accuracy of random forest classifier:', accuracy_xtest)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let’s use the cross_val_score() function to evaluate all classfier models\n",
    "using K-fold cross-validation, with five folds. Remember that K-fold cross\u0002validation means splitting the training set into K-folds (in this case,five), then mak‐\n",
    "ing predictions and evaluating them on each fold using a model trained on the\n",
    "remaining fold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.98571429, 0.98571429, 0.98571429, 0.99      , 0.98285714])"
      ]
     },
     "execution_count": 144,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    " from sklearn.model_selection import cross_val_score\n",
    "score= cross_val_score(rfc, X_train, Y_train,cv=5,scoring= \"accuracy\")\n",
    "score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Tune the random forest classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=5, estimator=RandomForestClassifier(),\n",
       "             param_grid=[{'max_features': [2, 4, 6, 8],\n",
       "                          'n_estimators': [3, 10, 30, 200]},\n",
       "                         {'bootstrap': [False], 'max_features': [2, 3, 4],\n",
       "                          'n_estimators': [3, 10, 30, 200]}],\n",
       "             scoring='accuracy')"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "param_grid = [\n",
    " {'n_estimators': [3, 10, 30,200], 'max_features': [2, 4, 6, 8]},\n",
    " {'bootstrap': [False], 'n_estimators': [3, 10,30,200], 'max_features': [2, 3, 4]},\n",
    " ]\n",
    "forest_clf = RandomForestClassifier()\n",
    "grid_search = GridSearchCV(forest_clf, param_grid, cv=5,\n",
    " scoring='accuracy')\n",
    "grid_search.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'max_features': 6, 'n_estimators': 200}"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# verifying the best parameter\n",
    "best= grid_search.best_params_\n",
    "best"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(max_features=6, n_estimators=200)"
      ]
     },
     "execution_count": 150,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#verifying the best estimator\n",
    "best_estimator= grid_search.best_estimator_\n",
    "best_estimator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 151,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Fitting  and predicting using the best parameters and the best estimator\n",
    "best_estimator.fit(X_train,Y_train)\n",
    "best_pred=best_estimator.predict(X_test)\n",
    "best_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the accuracy of random forest classifier: 0.9866666666666667\n"
     ]
    }
   ],
   "source": [
    "#verifying the accuracy\n",
    "best_accuracy= accuracy_score(Y_test,best_pred)\n",
    "print('the accuracy of random forest classifier:',best_accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The logistic Regression\n",
    "it is a linear model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(random_state=40)"
      ]
     },
     "execution_count": 156,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fitting The logistic Regression on the training set\n",
    "classifier= LogisticRegression(random_state= 40)\n",
    "classifier.fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Making prediction on the testing set\n",
    "pred_xtest1= classifier.predict(X_test)\n",
    "pred_xtest1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "**Check out the Performance of our classificatiom model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the accuracy of logistic regression: 0.956\n"
     ]
    }
   ],
   "source": [
    "# Check out the accuracy on the testing set\n",
    "Accuracy_logi= accuracy_score(Y_test, pred_xtest1)\n",
    "print('the accuracy of logistic regression:', Accuracy_logi)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.95142857, 0.94857143, 0.96142857, 0.95428571, 0.93285714])"
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#using the cross validation\n",
    "from sklearn.model_selection import cross_val_score\n",
    "logi_cross= cross_val_score(classifier, X_train, Y_train, cv=5,scoring=\"accuracy\")\n",
    "logi_cross"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Decision Tree Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(random_state=40)"
      ]
     },
     "execution_count": 162,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Fitting the Decision Tree classifier\n",
    "decision= DecisionTreeClassifier(random_state=40)\n",
    "decision.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 163,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#making prediction on the testing set data\n",
    "prediction= decision.predict(X_test)\n",
    "prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the accuracy of decision tree classifier: 0.9833333333333333\n"
     ]
    }
   ],
   "source": [
    "# Check out the accuracy on the testing set\n",
    "Accuracy_decision= accuracy_score(Y_test, prediction)\n",
    "print('the accuracy of decision tree classifier:', Accuracy_decision)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.97714286, 0.98428571, 0.98714286, 0.98857143, 0.98285714])"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "decision_cross= cross_val_score(decision, X_train, Y_train, cv=5,scoring=\"accuracy\")\n",
    "decision_cross"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##Tuning the decision treee classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(estimator=DecisionTreeClassifier(random_state=40),\n",
       "             param_grid={'max_depth': [None, 2, 4, 6, 8, 10]},\n",
       "             scoring='accuracy')"
      ]
     },
     "execution_count": 167,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "max_depth= [None, 2,4,6,8,10]\n",
    "parameters= {'max_depth': max_depth}\n",
    "deci_classifier= DecisionTreeClassifier(random_state= 40)\n",
    "grid= GridSearchCV(deci_classifier,parameters,scoring= 'accuracy')\n",
    "grid.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(random_state=40)"
      ]
     },
     "execution_count": 168,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#getting the best estimator\n",
    "deci_best_estimator= grid.best_estimator_\n",
    "deci_best_estimator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [],
   "source": [
    "deci_best_estimator.fit(X_train,Y_train)\n",
    "best_pred_deci= deci_best_estimator.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the accuracy of decision tree classifier: 0.9833333333333333\n"
     ]
    }
   ],
   "source": [
    "# Check out the accuracy on the testing set\n",
    "best_accuracy_decision= accuracy_score(Y_test,best_pred_deci)\n",
    "print('the accuracy of decision tree classifier:', best_accuracy_decision)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the accuracy for naive :0.89\n",
      "the accuracy for randomforestclassifier:0.98\n",
      "the accuracy for logisticregression :0.95\n",
      "the accuracy for decision tree:0.98\n",
      "the Stochastic Gradient Descent(SGD) classifier:0.95\n"
     ]
    }
   ],
   "source": [
    "# Compare the accuracy of the classse\n",
    "print(\"the accuracy for naive :0.89\")\n",
    "print(\"the accuracy for randomforestclassifier:0.98\")\n",
    "print(\"the accuracy for logisticregression :0.95\")\n",
    "print(\"the accuracy for decision tree:0.98\")\n",
    "print(\"the Stochastic Gradient Descent(SGD) classifier:0.95\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## As we can see there are 2 machine learning models have the same valor for their accuracy,in this case i choose the random forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x24ff59a2a00>"
      ]
     },
     "execution_count": 172,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting the confusion matrix\n",
    "cm= confusion_matrix(Y_test, best_pred)\n",
    "sns.heatmap(cm, annot=True, cbar= False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "The confusion matrix indicates that we have 1403+120 correct prediction and 17 + 4 incorrect prediction\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 0.],\n",
       "       [1., 0.],\n",
       "       [1., 0.],\n",
       "       ...,\n",
       "       [1., 0.],\n",
       "       [1., 0.],\n",
       "       [1., 0.]])"
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#predicting the probability for an employee turnover\n",
    "best_estimator.predict_proba(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Saving the best machine learning model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Appllying pickle to save the model\n",
    "import pickle\n",
    "data= {'model':best_estimator, 'scaler':standard}\n",
    "with open('randomforest.pkl', 'wb') as file:\n",
    "    pickle.dump(data,file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Loading the saving model\n",
    "with open('randomforest.pkl', 'rb') as file:\n",
    "    data= pickle.load(file)\n",
    "    \n",
    "loaded=data[\"model\"]\n",
    "loaded_scaling=data[\"scaler\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 176,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loaded.predict(X_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., ..., 0., 0., 0.])"
      ]
     },
     "execution_count": 179,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "proba1= loaded.predict_proba(X_test)\n",
    "proba=proba1[:, 1]\n",
    "proba"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [],
   "source": [
    "proba = Y_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Conclusion:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "After following step by step a lifecycle of a data science project,we found a lot of important information \n",
    "about customers' relationship with the Bank,we can suggest to Bank's marketing department to run a new campaign\n",
    "with a guarante of more than 90% of customers will accepte.\n",
    "\n",
    "In our EDA , we saw that Most of customer don't have a credit card in the bank\n",
    "*Most of customer don't have any CD Account in the bank.\n",
    "Bank must offer more credit card to the customers , and asking them to get their CD Account in the bank.\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
