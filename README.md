## Near Earth Asteroid Hazard/Size/Miss Predictor
We will use a dataset which compiles over 40,000 Near Earth Asteroids(NEA) and their characteristics to predict feature labels like Hazard Classification, and we will deploy these trained models to GCP as a prediction service app. By project's end, our GCP endpoint will serve predictions from three models for the following features:

-Hazard Classification: predict if an asteroid is potentially hazardous from orbital and physical features

-Miss Distance: predict Earth miss distance

-Size Estimation: predict diameter from absolute magnitude and albedo


The following is a high-level overview of the project's expected trajectory assuming we only train a model for the hazard classification task, along with each step's expected completion time in days:

#### Data Ingestion and Feature Engineering:
- load data set `near_earth_asteroids_2025.csv` as a pandas dataframe
- after dropping redundant features and being left with 14 `float` and 2 `str` columns: use scaling and one hot encoding during model training phase
    
 ### Model Training:
- due to the inbalanced target feature labels in `pha` we will use stratified k-fold cross-validation
- to find the best model, we will use GridSearchCV

**Timeline: 2 days(3 days if adding testing and logging)**

### Server File and Artifacts:
- The file server.py will contain server functionality and is where we'll load our saved model and feature artifacts. The server will be based on FastAPI and to validate incoming data we'll use Pydantic
- We'll implement three endpoints: "/" , "/predict", "/health"

**Timeline: 2 days**


### Docker and GCP deployment:
- The previous files will be containerized as a docker container and deployed to Google Cloud Platform

**Timeline: 2 days**


The above is a simple scaffolding for the prediction of a single feature and thus, the timelines are assuming a single model is being deployed. What we want are three models that predict three different features, meaning that all assumed timeline's may be doubled. **This gives the project an estimated time to completion of 14 days**. Of course, it may well be that some steps do not become much more complicated by the addition of other models, and so the estimated time to completion of 14 days  is considered a worst case scenario.

