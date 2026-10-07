# Trend Prediction Microservices

A microservices-based machine learning application for predicting Walmart sales trends using regression and classification models, with additional modules for Amazon product review analysis and recommendation systems.

The system combines data processing, machine learning, REST APIs, Docker containerization, and a web dashboard into a modular architecture.

---

## Architecture
The main system follows a microservices architecture:
Client/Web Dashboard  >  **API Gateway (8000)**  >  **Trend Service (8001)**  >  **Data Service (8002)**

### Services

#### API Gateway
The API Gateway represents the main entry point for client requests.
It:

- receives client requests;
- routes requests to the Trend Service;
- handles communication errors;
- provides access to the web dashboard.

#### Trend Service
The Trend Service contains the machine learning functionality.
It:

- trains regression models;
- trains classification models;
- generates sales predictions;
- calculates evaluation metrics;
- stores generated models and metrics locally during execution.

#### Data Service
The Data Service:
- loads the Walmart sales dataset;
- performs data preprocessing;
- performs feature engineering;
- creates historical sales features;
- provides processed data to the Trend Service.

---

## Web Dashboard
The project includes a web-based dashboard available through the API Gateway:

```text
http://localhost:8000
```

The dashboard provides a graphical interface for interacting with the machine learning system.

It allows users to:

- train Linear Regression and Random Forest models;
- train Decision Tree and XGBoost classification models;
- view model evaluation metrics;
- enter input features for sales prediction;
- receive predicted sales values;
- monitor the status of model training and prediction requests.

The dashboard is implemented as a static web interface served by the API Gateway.

---

## Project Structure

```text
trend_prediction_microservices/
│
├── api-gateway/
│   ├── app/
│   │   ├── static/
│   │   │   └── index.html
│   │   ├── main.py
│   │   └── routes.py
│   ├── .dockerignore
│   ├── Dockerfile
│   └── requirements.txt
│
├── data-service/
│   ├── app/
│   │   ├── data_provider.py
│   │   └── main.py
│   ├── .dockerignore
│   ├── Dockerfile
│   └── requirements.txt
│
├── trend-service/
│   ├── app/
│   │   ├── classification_trainer.py
│   │   ├── main.py
│   │   ├── model.py
│   │   └── trainer.py
│   ├── .dockerignore
│   ├── Dockerfile
│   └── requirements.txt
│
├── amazon/
│   ├── analysis/
│   ├── sentiment/
│   ├── collaborative_filtering/
│   └── content_based/
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Datasets
The project uses two main data sources.

### Walmart Sales Dataset
The Walmart dataset is used for sales trend prediction and popularity classification.
The main columns include:

- `Store`
- `Dept`
- `Date`
- `Weekly_Sales`
- `IsHoliday`

The complete dataset is stored locally and is excluded from the GitHub repository because of its size.

### Amazon Electronics Reviews Dataset
The Amazon Electronics Reviews dataset is used for:
- review analysis;
- sentiment analysis;
- collaborative filtering;
- content-based recommendation.

The complete Amazon dataset is also stored locally and is excluded from the GitHub repository because of its large size.

---

## Feature Engineering

The Data Service extracts temporal characteristics from the `Date` column:

- `year` - year of the observation;
- `month` - month of the observation;
- `week_of_year` - week of the year;
- `day_of_week` - day of the week.

Historical sales features are calculated separately for each store and department:

- `sales_lag_1` - sales from the previous week;
- `sales_lag_2` - sales from two weeks earlier;
- `sales_previous_7_weeks` - average sales from the previous seven weeks.

Historical features are calculated using previous observations only, so the current target value is not included in the historical averages.

---

## Machine Learning Models

The Walmart part of the system implements four machine learning models.

### Regression

#### Linear Regression
Linear Regression is used as a baseline regression model for predicting weekly sales.

#### Random Forest Regression
Random Forest Regression is used to model non-linear relationships between input characteristics and weekly sales.

The final configuration uses:

```text
n_estimators = 100
random_state = 42
n_jobs = 1
```

### Classification
The classification task predicts whether a product belongs to the defined popular-product category.

#### Decision Tree
Decision Tree is used as an interpretable classification model for predicting product popularity.

#### XGBoost
XGBoost is used as an ensemble classification model based on gradient boosting.

---

## Train-Test Split

The Walmart dataset is divided into chronological training and testing subsets.

The earliest 80% of observations are used for training, while the most recent 20% are used for testing.

A chronological split is used because the system predicts future sales based on historical observations.

For the final experiments:

```text
Training samples: 329,168
Testing samples: 85,777
```

Training period:

```text
2010-02-19 - 2012-04-06
```

Testing period:

```text
2012-04-13 - 2012-10-26
```

---

## Model Evaluation

### Regression Metrics

The regression models are evaluated using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² (Coefficient of Determination)**

MAE measures the average absolute difference between predicted and actual values.

RMSE gives greater weight to larger prediction errors.

R² measures the proportion of variance in the target variable explained by the model.

### Classification Metrics

The classification models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion matrix**

---

## Amazon Review Analysis

The project contains additional modules for analyzing Amazon Electronics product reviews.

### Dataset Analysis

The analysis module processes the Amazon dataset line by line and calculates descriptive statistics such as:

- number of reviews;
- number of users;
- number of products;
- rating distribution;
- reviews containing text;
- time range of the dataset.

### Sentiment Analysis

The sentiment analysis module classifies selected reviews into positive and negative categories.

The implementation uses:

- TF-IDF text representation;
- Logistic Regression;
- stratified train-test split.

The model is evaluated using:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- confusion matrix.

### Collaborative Filtering

The collaborative filtering module uses user-item interactions to generate product recommendations.

The system uses item co-occurrence and cosine similarity to identify related products.

The recommendation system is evaluated using:

- Precision@10;
- Recall@10;
- Hit Rate@10.

### Content-Based Filtering

The content-based recommendation module uses product review text to calculate similarities between products.

TF-IDF is used for text representation, while cosine similarity is used to measure similarity between product representations.

The recommendation system is evaluated using:

- Precision@10;
- Recall@10;
- Hit Rate@10.

---

## Docker

The complete microservices architecture can be started using Docker Compose.

The project contains three Dockerized services:

```text
API Gateway
Trend Service
Data Service
```

From the project root directory, run:

```bash
docker compose up --build
```

---

## Available Services

API Gateway:

```text
http://localhost:8000
```

Trend Service:

```text
http://localhost:8001
```

Data Service:

```text
http://localhost:8002
```

The web dashboard is available through:

```text
http://localhost:8000
```

---

## Swagger Documentation

The REST APIs can be explored using Swagger UI:

```text
http://localhost:8000/docs
http://localhost:8001/docs
http://localhost:8002/docs
```

---

## Main API Endpoints

### Regression

Train a regression model:

```text
POST /modeli/treniraj
```

Return regression metrics:

```text
GET /modeli/metrike
```

Generate a sales prediction:

```text
POST /modeli/predvidi
```

Supported regression models:

```text
linear
rf
```

### Classification

Train a classification model:

```text
POST /klasifikacija/treniraj
```

Return classification metrics:

```text
GET /klasifikacija/metrike
```

Supported classification models:

```text
decision_tree
xgboost
```

---

## Example Request

Train the Random Forest regression model:

```text
POST http://localhost:8000/modeli/treniraj?model=rf
```

Example prediction request:

```json
{
    "store": 1,
    "dept": 1,
    "is_holiday": 0,
    "year": 2012,
    "month": 10,
    "week_of_year": 42,
    "day_of_week": 4,
    "sales_lag_1": 16000,
    "sales_lag_2": 15500,
    "sales_previous_7_weeks": 15800
}
```

Example response:

```json
{
    "predicted_sales": 16125.42
}
```

The numerical value above is only an example of the response format and does not represent a final experimental result.

---

## Generated Model Files

During model training, the Trend Service generates model and metric files locally.

Examples include:

```text
app/model.joblib
app/metrics.joblib
app/classification_models.joblib
app/classification_metrics.joblib
```

These generated files are intentionally excluded from the GitHub repository using `.gitignore` and are not required for storing the source code.

Large trained model files can require significant storage and computational resources.

---

## Git and Data Management

The repository uses `.gitignore` to exclude:

- large datasets;
- generated model files;
- generated CSV files;
- Python cache files;
- local virtual environments;
- environment configuration files.

The datasets remain available locally for running the experiments but are not included in the public GitHub repository.

---

## Technologies

- Python
- FastAPI
- Docker
- Docker Compose
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Requests
- Git
- GitHub
- HTML
- CSS
- JavaScript

---

## Skills Demonstrated

- Microservices Architecture
- Machine Learning
- Regression
- Classification
- Feature Engineering
- REST API Development
- FastAPI
- Docker Containerization
- Model Training
- Model Evaluation
- Data Processing
- Sentiment Analysis
- TF-IDF
- Recommendation Systems
- Collaborative Filtering
- Content-Based Filtering
- API Gateway Design
- Web Dashboard Development

---

## Future Improvements

Possible future improvements include:

- model versioning;
- hyperparameter optimization;
- additional machine learning models;
- authentication and authorization;
- CI/CD pipeline;
- improved monitoring and logging;
- Kubernetes deployment;
- cloud deployment;
- automated model retraining.

---

## Author

Mina Lazović