# Trend Prediction Microservices

A microservices-based machine learning application for predicting Walmart sales trends using FastAPI, Docker, and Scikit-learn.

The project follows a microservices architecture, where data processing, machine learning, and API logic are separated into independent services. The API Gateway acts as a single entry point for client requests.

The project also contains modules for analyzing Amazon product reviews, including sentiment analysis and recommendation systems.

---

## Architecture

Client > **API Gateway (8000)** > **Trend Service (8001)** > **Data Service (8002)**

### Services

- **API Gateway**
  - Receives client requests
  - Routes requests to the appropriate microservice
  - Handles communication errors

- **Data Service**
  - Loads the Walmart dataset
  - Performs feature engineering
  - Prepares historical sales features
  - Returns processed data

- **Trend Service**
  - Trains machine learning models
  - Stores the active model
  - Generates predictions
  - Returns evaluation metrics

---

## Project Structure

    project/

    │── api-gateway/
    │── trend-service/
    │── data-service/
    │── amazon/
    │── docker-compose.yml
    │── .gitignore
    │── README.md

---

## Dataset

The project uses the Walmart Sales dataset (`train.csv`) containing the following columns:

- `Store`
- `Dept`
- `Date`
- `Weekly_Sales`
- `IsHoliday`

The project also uses the Amazon Electronics Reviews dataset (`Electronics_5.json.gz`) for:

- review analysis;
- sentiment analysis;
- collaborative filtering;
- content-based filtering.

The complete Amazon dataset is stored locally because of its large size.

---

## Technologies

- Python
- FastAPI
- Docker
- Docker Compose
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Requests
- SQLite
- Git
- GitHub

---

## Feature Engineering

The Data Service extracts the following features from the `Date` column:

- `year` - year of the observation
- `month` - month of the observation
- `week_of_year` - week of the year
- `day_of_week` - day of the week

Historical sales features are also created for each store and department:

- `sales_lag_1` - sales from the previous week
- `sales_lag_2` - sales from two weeks earlier
- `sales_previous_7_weeks` - average sales from the previous seven weeks

Historical features are calculated using previous observations only.

---

## Machine Learning Models

The project implements two regression models:

- **Linear Regression (`linear`)**
  - Baseline regression model used for comparison.

- **Random Forest Regressor (`rf`)**
  - Ensemble model capable of capturing non-linear relationships.

The Random Forest model uses:

- `n_estimators = 200`
- `random_state = 42`

---

## Train-Test Split

The dataset is divided chronologically into training and testing subsets.

The earliest 80% of observations are used for training, while the most recent 20% are used for testing.

A chronological split is used because the system predicts future sales based on historical data.

---

## Model Evaluation

The trained models are evaluated using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² (Coefficient of Determination)**

MAE measures the average absolute prediction error.

RMSE gives greater weight to larger prediction errors.

R² measures the proportion of variance in the target variable explained by the model.

---

## Amazon Review Analysis

The project contains additional modules for Amazon product review analysis.

### Sentiment Analysis

The sentiment analysis module classifies selected reviews into positive and negative categories.

The implementation uses TF-IDF text representation and machine learning classification.

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Collaborative Filtering

The collaborative filtering module uses user-item interactions to generate product recommendations.

The system is evaluated using:

- Precision@10
- Recall@10
- Hit Rate@10

### Content-Based Filtering

The content-based recommendation module uses product review text to calculate similarities between products.

TF-IDF is used for text representation and cosine similarity is used to measure product similarity.

The system is evaluated using:

- Precision@10
- Recall@10
- Hit Rate@10

---

## Running the Project

### Prerequisites

Make sure Docker Desktop is installed.

### Build and Start

From the project's root directory run:

    docker compose up --build

---

## Available Services

API Gateway: http://localhost:8000

Trend Service: http://localhost:8001

Data Service: http://localhost:8002

---

## Swagger Documentation

The REST APIs can be explored using Swagger UI:

- http://localhost:8000/docs
- http://localhost:8001/docs
- http://localhost:8002/docs

---

## Main API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/modeli/treniraj` | Train a machine learning model |
| GET | `/modeli/metrike` | Return model evaluation metrics |
| POST | `/modeli/predvidi` | Generate sales predictions |

Supported models:

- `linear`
- `rf`

---

## Example Request

Train the Random Forest model:

    POST http://localhost:8000/modeli/treniraj?model=rf

Example prediction request:

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

Example response:

    {
        "predicted_sales": 16125.42
    }

The numerical value above is only an example of the response format and does not represent a final experimental result.

---

## Model Files

After training, the Trend Service stores:

- `app/model.joblib`
- `app/metrics.joblib`

The stored metrics include:

- MAE
- RMSE
- R²
- training samples
- test samples
- training period
- testing period

---

## Skills Demonstrated

- Microservices Architecture
- Machine Learning
- Regression
- Feature Engineering
- REST API Development
- FastAPI
- Docker Containerization
- Model Training
- Model Evaluation
- Data Processing
- Sentiment Analysis
- Recommendation Systems
- Collaborative Filtering
- Content-Based Filtering

---

## Future Improvements

- Add model versioning
- Hyperparameter optimization
- Add additional machine learning models
- Integrate a database
- Implement authentication
- Add CI/CD pipeline
- Deploy with Kubernetes
- Cloud deployment
- Improve monitoring and logging

---

## Author

Mina Lazović