# Predicting Flight Delays to Mitigate Potential Delays and Costs

## Project Abstract

Airline On-Time Performance, defined as a flight arriving within 15 minutes of its scheduled arrival time, is a critical metric for airlines and regulators. Analysis of TranStats flight data from the US Department of Transportation (USDOT) reveals that 18% of flights were delayed by more than 15 minutes between 2015 and 2021. Delays incur substantial operational costs, especially with the USDOT’s ruling requiring automatic refunds for domestic flights delayed by three hours or more. To address this, our project focuses on predicting flight delays two hours before departure, with a primary goal of minimizing false negatives to preserve trust and prevent customer churn.

Our pipeline incorporates flight data from TranStats and weather data from NOAA to build a robust machine learning model, excluding delay causes like mechanical issues or staffing shortages to maintain practical generalizability. Baseline logistic regression models achieved an F-Beta test score of 0.699 with 21 features, improving to 0.714 when 6 custom features—such as previous flight delays and cyclical time—were added. Experiments with random forests, bagging, boosting, and neural networks identified random forests as the best-performing model, achieving superior accuracy and an F-Beta score of 0.765 on the test set.

Our random forest model demonstrated the best ability to capture complex relationships in the data while maintaining interpretability, making them ideal for predicting delays in real-world scenarios. This model resulted in a 0.768 F-beta score on unseen 2019 test data. Our model trained on 2015-2018 also showed strong generalizability when testing on 2020 data with a F-beta score of 0.776. While our initial model shows strong promise in predicting delays, we see opportunities of improvement by incorporating additional features and trying ensembling to further optimize the model for operational use across the airline industry.

## Data


### Description of Data

The data for the project will be sourced from three different datasets.
  - The first is flight information from the US Department of Transportation (DOT). This contains 109 features and ~31.7 million rows. It contains information related to the flights such as departure and arrival destinations, fligth durations (taxi and flight times), carrier information, distance traveled, and whether the flight was delayed or diverted. This data will be limited to US states and territories for depature and arrival locations. The full dataset contains information from 2015 to 2021 and totals 2.94 GB.
  - The second dataset contains weather information from the National Oceanic and Atmospheric Administration. This contains 177 features and ~630.9 million rows. It contains various weather information such as temperature, humidity, precipitation, visiblity, sunrise time, and sunset time. The location information and time information can be used to join with the flight dataset to gain information on potential weather features impacting flight delays. The full dataset contains information from 2015 to 2021 and totals 35.05 GB.
  - The final dataset contains airport information from the US DOT. This contains 10 features and ~18K rows. It contains location information for airports which can be used to merge the weather and flight datasets.

The data has been split into various time intervals to perform initial model development. The rolling window for time splits was set up to cover 24-hour periods, allowing us to incrementally shift the training and test sets to simulate real-time prediction scenarios. This approach ensured that each test set contained data from a subsequent day, providing a realistic evaluation of model performance over time. This approach involved splitting the dataset into training and test sets based on time, ensuring that the model was trained on past data and tested on future data. This method closely mimics real-world scenarios where predictions need to be made based on historical information. By using rolling windows, we also ensured that the model could adapt to seasonal variations and other temporal patterns that affect flight delays.

We masked the data by setting values to None if the twoHoursPriorDiffPrevF value was less than or equal to zero. This ensured that data from previous flights was only retained if it was relevant for predicting current flight conditions, effectively maintaining data quality and relevance. Initially, we filtered the dataset to focus only on flights originating and landing within the United States, as international flights often have additional complexities related to customs and air traffic control regulations. We also filtered out records with missing or erroneous values, particularly those related to departure and arrival times, which are crucial for predicting delays.

### Data Dictionary
The data from each source has been narrowed down to useful features. Please see the data dictionary to see the complete list of features for each dataset: [(Data Dictionary)](https://docs.google.com/spreadsheets/d/1cxMpgoy3YIUD1OGv9_BM3s-Q6DRTpuuF_pKyUKDrXJc/edit?gid=0#gid=0).

Key data elements to support our model prediction are listed below:
| Data Element | Objective |
| ----- | ----- |
| <b>Flight Data</b> | |
| Reporting Airline and Flight Number | Identifies a flight route with airline company between airports |
| Tail Number | Identifies an airline enabling the reconstruction of a plane's flight history |
| City, State, Latitude, Longitude | Filter data to US and US territories and connect with weather data |
| Destination and Arrival Airport | Basis to create graph features to measure an airport's influence on flight network |
| <b> Prediction Objective </b> | |
| Departure Delay Indicator | Primary prediction: boolean indicator if a departure was delayed 15 minutes or more |




## Machine Learning Pipeline

Our current machine learning pipeline and checkpointing strategy is as follows:
![pipeline1](https://github.com/gilbertwong22/mids_w261_final_project/blob/main/datapipe1.png?raw=true)
![pipeline2](https://github.com/gilbertwong22/mids_w261_final_project/blob/main/datapipe2.png?raw=true)

We begin our EDA and feature engineering on a subset of 3-month and 1-year flight data prior to expanding to the entire dataset. To ensure modularity, rapid prototyping, and failsafes against disruptions, we used these checkpoints up to model training:
- <b> Custom Data Model </b> - Initial basis after processing data for further feature engineering and null handling.
- <b> Feature Engineering </b> - Adding derived features and undersampling non-delayed flights to enhance our model's predictive power. We complete another round of masking to prevent leakage and address our remaining nulls.
- <b> Encoding </b> - Final transformations such as scaling and indexing to prep our data for model training. The encoding is fit on the training set and applied to both the train and test set to ensure our model can maintain genearlizability on unseen data in the test set. 
- <b> Model Training </b> - Model architecture, callbacks, early stopping, and model saving are employed to guard against training disruptions and provide a quick restart to downstream predictions and metric analyses.

Cross validation is used as a grid search method to fine-tune the pipeline.  The 3 folds cross validation datasets is used to find the best hyper-parameters for the models on hand.  We use a weighted average of [0.2, 0.3, 0.5] to more heavily weigh our model on recent results. The cross validation for hyperparmeters tuning results is shown below:

| Model | Best Hyperparameters Selected |
|----------|----------|
| Decision Tree   | depth=6, min_info_gain=0.01   |
| Logistic Regression    | regParam=0.1, elasticNetParam=0.0, threshold=0.4   |
| Neural Network    | 3 layers, maxIter = 100   |
| Random Forrest    | depth = 10, numTrees = 5   |
| Gradient Boost    | depth = 10, maxIter = 10  |

Based on the models below, we found the top 10 important features that could affect the model performance include:
- Previous flight related (ie. Previous flight delayed, previous flight delay group, previous flight departure time, previous flight distance, time between flights)
- Weather (ie. hourly precipitation, hourly visibility, hourly wind gust speed)
- Own flight information (ie. departure time, origin) 


### Initial EDA

The data was explored to gain an understanding of which features could potentially be used, potential feature correlation, and data distribution.
  - The project required use of US state and territory flights (departure and arrival). The flights dataset was reviewed and only contains US state and territory depatures and arrivals.
  - It was determined that each value in the dataset was duplicated. This was likely due to a departure and arrival record for each flight. Each dataset was deduplicated before proceeding.
  - Upon reviewing the data it was determined the predicted outcome variable (delayed more than 15 minutes) is skewed, 82% of the flights were on time and 18% were delayed. This makes sense as most flights aren't delayed. Models and evaluation metrics will need to account for this skew in the outcome variable.
  - There was also a right skew in the data for the delay times. The average delay time was 9.2 minutes but the median was -2.0 minutes (early). The minimum delay time was -29 minutes and the maximum was 1,175 minutes.







![SeasonalityFlightCount](https://github.com//ngasserberk/mids-w261-final_project/blob/main/delay_dist.png?raw=true)

  - To gain an understanding if there was a uniform distribution of flights, the count of flights and percentage of delayed flights by year, month, and day of week were reviewed on a sample of the full dataset (2015-2021). 
    - In 2018 and 2019, the count of flights began to increase before drastically decreasing in 2020 and 2021, likely due to COVID-19 pandemic.
    - There was a smaller number of delayed flights in 2020 compared to other years (2021 was similar to 2015-2019).
    - There appears to be seasonality across months of the data. Similar behavior occurs throughout the week, with a lower percent of delays on Monday and Tuesday. Finally, there is a seasonality component based on the hour of day the flight departs.

![SeasonalityFlightCount](https://github.com//ngasserberk/mids-w261-final_project/blob/main/seasonality_flight_count.png?raw=true)

![SeasonalityFlightPercent](https://github.com//ngasserberk/mids-w261-final_project/blob/main/seasonality_delay_perc.png?raw=true)

![SeasonalityDepTime](https://github.com//ngasserberk/mids-w261-final_project/blob/main/percent_delay_dep_time.png?raw=true)

  - The distribution of flights by airlines was reviewed and the distribution of delay time from the full dataset (2015-2021).
    - The most flights are with WN. The majority of the flights are by WN, DL, AA, OO, and UA. Following those, there are 15 other airlines with fewer flight counts.
    - There does not seem to be a destinct correlation between number of flights for an airline and delay time. As shown earlier, there is a large skew in the delay times for each airline. The figure was restricted to a maximum delay time of 100 minutes while we saw earlier the max delay was 1,175 minutes.
    - Airlines F9, B6, QX, and WN appear to have the widest distribution of delay times.

![FlightsByAirline](https://github.com//ngasserberk/mids-w261-final_project/blob/main/flights_by_airline.png?raw=true)

![AirlineBoxPlot](https://github.com//ngasserberk/mids-w261-final_project/blob/main/airline_box_delays.png?raw=true)

### Custom Data Model
Our custom data model with essential joins and features is displayed below. Reference the Final Data Dictionary for a complete list of columns: [Data Dictionary](https://docs.google.com/spreadsheets/d/1cxMpgoy3YIUD1OGv9_BM3s-Q6DRTpuuF_pKyUKDrXJc/edit?gid=896383382#gid=896383382)

![AirlineBoxPlot](https://github.com/jasondongmids/mids_w261_final_project/blob/main/ref/custom_data_model_er_diagram.png?raw=true)

- **Timezone:** We use latitude and longitude and the `pytz` package to find the timezone associated with each airport and weather station. The timezone is then used to convert features from local to universal time (UTC).
- **Previous Flights**: Previous flights were joined onto current flights by joining on the tail number, carrier, and origin airport id of the current flight to destination airport id of the previous flight. At this stage, we do an initial check if the departure time was after the arrival time of the previous flight. The previous flights that were merged were ranked and the latest flight kept. We perform another round of masking by comparing if the departure or arrival time is within the two hour window before scheduled departure time. If yes, we remove the relevant departure or arrival data of the previous flight.
- **Weather Features**: We filter weather data to only include US stations and hourly readings to reduce the size of our data. We find closest weather station associated with each airport from the stations and airport codes data. After converting weather measurement times to UTC, we join our hourly weather data to flight information by finding the most recent weather reading between four and two hours before scheduled departure time. The two hour window weather measurements provided the optimal balance between processing times and data population with over 99.9% of flights associated with station data after the join.

PageRank, PROPHET, and other custom features from feature engineering are discussed further below.


### Missing Data 

Missing data at the feature level was reviewed. The count and percent of non-null values were reviewed for each dataset
  1. Initial quick analysis of the flights dataset was reduced to remove any feature that had less than 15% percent of non-null values. While, normally you wouldn't want to include features that sparse, some are only filled for delayed flights, such as the delay indicators (weather, carrier, etc.). Thus, these are only ~18% filled. The remaining features with less than 15% filled values were dropped, 48. This included flight information such as grounded time away from gate, and flight deviation information.
  2. The weather dataset was reviewed and determined to be sparse for the majority of the features. However, this dataset is at the latitude and longitude level. Thus, taking a direct reduction from the full table wouldn't make sense as some coordinates won't join with the flight data. Our strategy to address missing weather values is detailed below.

We addressed missing data in the following scenarios:
- <b> Categorical Data </b> - Data specific to delayed flights will be filled with a generic value for on time flights. If there is not a generic value (ie delay) the value was filled with -999 to represent an unknown value.
- <b> Previous Flight Data </b> - We populate planned departure and arrival information for previous flights when the actual values are masked due to being within the two hour departure window.
- <b> Weather Data </b> - Missing weather data was filled in successively using the following strategies.
  - Linear interpolation was fist applied to missing values, if non-null values were within a two hour window
  - City level 6 hour rolling average was applied second to remaining missing values
  - Next, State level 6 hour rolling average was applied
  - If there was still missing values, mean values across the dataset were added.
  - The figure below shows the percent of filled values at each step of the process.

![weatherfill](https://github.com//ngasserberk/mids-w261-final_project/blob/main/weather_datafill.png?raw=true)

## Feature Engineering

To further improve our model we performed feature engineering to clean existing data and generate new features which could be useful to the model.

We conducted categorical feature engineering to convert categorical variables into a format suitable for machine learning models. For features like airline carrier codes, airport codes, and flight origins and destinations, we used String Indexing. String Indexing assigns a numeric index to each category, while One-Hot Encoding creates binary vectors representing each category. These transformations are essential for enabling machine learning models to understand and leverage categorical information. For instance, identifying which airline or airport is associated with higher delays can help us enhance prediction accuracy.

The main feature engineering that was performed was merging on the previous flight information. This was achieved by merging based on the plane’s tail number and airline. Then, the merged values were partitioned and sorted to keep the most recent previous flight information. To avoid data leakage during training, if the previous flight did not leave within two hours of the current flights planned departure time, all previous flight information was masked.

Time is a cyclic feature. After 23:59 it doesn’t go to 24:00 but 00:00. Thus, sinusoidal transformation to the scheduled departure time, day of week, and month to capture the cyclic and seasonality trends of time.

### Weather Forecasting

To further enhance the predictive power of our model, we decided to include weather features for our future models. As weather information around the time of actual flight departure will not be available at prediction time, using actual weather data around flight departure time would constitute data leakage. Hence, we utilized Facebook's Prophet time-series model to forecast weather variables such as precipitation, visibility, wind speed, wind gust speed, wet bulb temperature, dry bulb temperature, and dew point temperature, all of which will be used as forecast features for our predictive models. Through our time-series plots, we were able to see that Prophet accurately captured the seasonalities and trends of all the variables.

### PageRank
Flight delays in the airport of origin will have downstream effects for the timely departure of subsequent flights in the destination airports. We decided to use the PageRank algorithm, developed by Google to rank web pages, to measure the influence of airports and applied them in our models. Using the 12 months 2015 traffic data which contains over 5.6 million air flights with 4,675 unique routes and 322 airports in U.S, we created a graph network using airports as the nodes and the flight route between two airports as the edges. Our initial results seen below match our intuition as we can see major airport hubs are the most influential. Certain airports such as JFK are notably missing, potentially due to the fact that we're only focusing on domestic flights routes. 

The top 10 most important airports with the highest page rank scores are shown below:

| Rank | Airport |
|----------|----------|
| 1.    | Hartsfield-Jackson Atlanta International Airport   |
| 2.    | Chicago O'Hare International Airport   |
| 3.    | Dallas Fort Worth International Airport   |
| 4.    | Denver International Airport   |
| 5.    | Minneapolis–Saint Paul International Airport   |
| 6.    | George Bush Intercontinental Airport   |
| 7.    | Detroit Metropolitan Wayne County Airport   |
| 8.    | Salt Lake City International Airport   |
| 9.    | San Francisco International Airport   |
| 10.   | Los Angeles International Airport  |

![weatherfill](https://github.com/gilbertwong22/mids_w261_final_project/blob/main/pagerank%20airports%20map.png?raw=true)

## Data Balancing 

Most flights are on time, with only a small percentage experiencing significant delays. This imbalance can lead to biased model predictions, where the model becomes too focused on predicting the majority class (on-time flights) and fails to correctly identify delayed flights. To counteract this, we used undersampling to reduce the number of on-time flights to match the number of delayed flights. This balanced dataset helps the model learn more effectively about the factors contributing to delays, reducing the likelihood of it simply predicting that all flights are on time.

**Balance in Model Training**: By using a 50/50 ratio, we ensure that the model receives an equal number of delayed and on-time flights during training. This balance helps the model learn to recognize the factors associated with both classes more effectively, instead of being biased toward predicting the majority class (on-time flights). If we trained on the imbalanced dataset, the model would likely predict most flights as on-time, resulting in poor recall for the delayed class.

**Improving Recall for Delays**: In the context of flight delays, false negatives (i.e., predicting a flight will be on time when it is delayed) are more critical than false positives. With a balanced dataset, the model has an improved chance of detecting actual delays, thus increasing recall for the minority class. This is important because failing to predict a delayed flight can lead to operational disruptions and poor passenger experience.

**Model Generalizability**: Although undersampling reduces the number of on-time flights in the training set, it allows the model to focus more on learning the characteristics of delayed flights. This helps create a more generalizable model that performs well for both classes, rather than simply overfitting to the majority class. It ensures that the model does not always favor the majority (on-time flights), which would result in poor predictive performance for delayed flights.

## Data Leakage

Data leakage in machine learning is when you train a model on information that wouldn't be available if you were to make the prediction in real time.

With the delayed flight timeseries model, predictions need to occur two hours prior to the flights depature. Thus, a major data leakage issue would be any information that wouldn't be available until less than two hours prior to the flight. If the previous flight arrived thirty minutes prior to the current flight, arrival information can't be used in model training.
 - Our team mitigated this risk by first converting flight time information to UTC. This allowed us to ensure all times were on the same scale. Then, previous flight information was masked appropriately. If the previous flight departure time was within two hours, departure and arrival metrics were masked. If the departure time was more than two hours but arrival time was within two hours, the arrival metrics were masked.
 - Hourly merged weather data was merged on with weather readings two hours or more prior to the scheduled depature time. As discussed above, missing values for weather were filled in steps. Each step was using data the occurred two hours or more prior to the depature time. However, even using this method some data was still filled using mean values. This was a known, minor risk as twelve of the weather features had more than 99% of the values filled. Hourly pressure change required 4% of the values be filled with mean and hourly wind gust required 15% be filled with mean.
 - PageRank metric was calculated using data from 2015. Thus, using PageRank while training and testing on data within 2015 was data leakage. However, due to the limited data we had we felt it was the best approach to use 2015 data for PageRank and model testing. The use of PageRank in our production model and validation in 2019 would not be leakage as the data is known during that time.

Another source of data leakage in timeseries occurs when the data isn't sorted or split properly. If the model is trained on data the occurred after the testing data, this would be data leakage.
- This was mitigated using our train/test splits for each experiment, as defined below.

Data leakage would also occur if final model decisions are made using final held out validation data. Machine learning data should be split into three groups: train, test, and validation. To appropriately determine how well a model will perform in normal application, final model metrics should be calculated on data that wasn't used at all during training or testing.
 - This was accomplished by only training and testing models on data from 2015 to 2018. Data from 2019 was only used once, on the final production model. 

## Machine Learning Algorithms and Metrics
For this project, we are focusing on predicting **departure delays** where a delay is defined as being 15 minutes or greater past the planned departure time. The prediction will be made at least **two hours before departure** to allow sufficient time for airlines and airports to notify passengers and adjust operations. Our primary stakeholders include airlines, airports, and passengers. This is framed as a **classification problem**, where the target variable is whether a flight will be delayed or not.


### Metrics
Our model is targeted towards helping airline companies predict delays to minimize costs and increase on-time performance performance. Incorrect predictions can lead to the following outcomes for the business:

| <div style='width:150px'> False Positive </div> | <div style='width:290px'> False Negative </div>|
| ----- | ----- |
| Incorrectly identifying flight delays may lead to unneeded resource allocation for flights actually on-time | Delay will be missed from the model and lead to inaction on delays that could've been mitigated. |

While false positives are detrimental, we believe false negatives would be worse as the business would never be notified of a missing flight delay prediction. A predicted delay doesn't require immediate action. Decision makers can be supplied context for triage and feasibility of mitigating the delay before action is taken on the data. Due to these reasons, we will use the following metrics, in relation to the delayed class, to evaluate our models:

**Primary Metric**

1. **F-beta score**:
   $$
   \text{F-beta Score} = ((1 + \beta^2)) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times\text{Precision} + \text{Recall}}
   $$
   F-beta scores provide a weighted balance between precision and recall ensuring a balance between optimizing to minimize against false positives vs negatives. Due to our desire to capture more delays at the risk of false positives (ie. reduce false negatives), we want our f-beta score to prioritize recall. We choose a weight \\(\beta = 2\\) to weight the recall metric twice as heavily as precision.

**Sub-metrics**

2. **Recall**:
   $$
   \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
   $$
   Essential for broadly identifying the delays to minimize unexpected delays. Higher recall may lead to incorrectly identifying flight delays.

3. **Precision**:
   $$
   \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
   $$
   Indicates how many of our predicted delays are actual delays to avoid false alerts to airline companies. Higher precision may lead to missed predictions.


### Models
Our baseline and model iterations to predict departure delays will focus on the following algorithms:

1. **Logistic Regression**:
   - **Implementation**: Using `PySpark`'s `LogisticRegression` class.
   - **Loss Function**: Binary Cross-Entropy Loss:
     $$
     L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
     $$
   - **Reasoning**: A simple baseline to understand feature importance and build interpretability.

2. **Decision Tree**:
   - **Implementation**: Using `PySpark`'s `DecisionTreeClassifier`.
   - **Reasoning**: Simple yet powerful model requiring minimal feature engineering while providing high interpretability.

3. **Random Forest Classifier**:
   - **Implementation**: Using `PySpark`'s `RandomForestClassifier`.
   - **Feature Importance**: Helps identify critical factors contributing to delays.
   - **Advantage**: Good for capturing non-linear relationships and robust to overfitting with proper tuning while providing high interpretability.

4. **Gradient Boosted Trees**:
   - **Implementation**: Using `PySpark`'s `SparkXGBRegressor`
   - **Loss Function**: Logistic loss for binary classification.
   - **Advantage**: Effective for handling imbalanced data and complex relationships.
   - **Reasoning**: Provides high predictive power while maintaining efficiency.

5. **Neural Network**: 
   - **Implementation**: Using Keras, two hidden layers, 256 neurons each, ReLU activation function for non-linearity in both layers, dropout rate of 30% after each layer for regularization. Output layer with a single neuron using a sigmoid activation function for binary classification.  
   - **Loss Function**: Binary cross-entropy
   - **Reasoning**: Neural Netowrks are capable of capturing complex, non-linear relationships between many interdependent features.

6. **Distributed Deep Learning Model**: 
   - **Implementation**: Using PySpark.linalg, processed PySpark DataFrames by converting array<double> to Vectors using PySpark's VectorUDT. Training and labels extracted from PySpark DataFrames as NumPy arrays to be compatible with Keras. Fully distributed data handling via Spark; training itself runs on local TensorFlow/Keras.
   - **Loss Function**: Binary cross-entropy
   - **Reasoning**: Deep learning models ensure scalability and efficiency when working with such massive datasets, we wanted to experiment with this idea. 
![](path)

## Experiment Approach / Data Splits

 The temporal nature of the on time performance data requires special attention in splitting data during training and testing in order to maintain the time dependency between observations and to prevent data leakage of future values being used to predict the past. Model testing was divided into two phases, one year and four year training/testing.

1. **Single Year (2015)** 
  - Train: 9 months
  - Test: 3 months
  - The initial experiments were used to get a baseline understanding of model performance as well as determine what subset of features performed best before expanding to more data
  - The same baseline features were used for each model for initial testing to generate comparable baseline results across each model

![1yeardata](https://github.com//ngasserberk/mids-w261-final_project/blob/main/1year_datasplit.png?raw=true)

2. **Four Year (2015-2018) Cross-Validation**
  - A rolling blocked cross-validation strategy was used to iterate over the entire dataset (2015-2018)
   - Train: 15 months to ensure seasonality effects are included while training the model 
   - Test: 3 months
   - Each block had an overlap of 1 week to ensure continuity betwen blocks.
   - The cross-validation was used to perform hyperparameter tuning. A grid search approach was taken with each model to test various model parameters.
   - A weighted average of model metrics from cross-validation split was calculated. A weight of 0.2 was given to the first split (oldest data), 0.3 to the second split, and 0.5 to the most recent data. This approach weighted more recent data over older data.
   
![xvaldata](https://github.com//ngasserberk/mids-w261-final_project/blob/main/xval_datasplit.png?raw=true)

3. **Four Year (2015-2018)**
  - The best performing model parameters were trained across the whole four year dataset.

4. **Final Validation (2019)**
  - The most recent data (2019) was held out from all training and testing. It was only used on the final chosen model after four year testing.
  - Final model performance was based on prediction results from 2019.

The experiments were trained on 56GB, 16 core driver and 84GB, 24 core workers.

## Results

#### Logistic Regression
###### Train and Test Results for delayed flights with 2015 Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| Baseline Model <br> (31 Features) | 2 min | Train: 0.6799 <br> Test: 0.7309 | Train: 0.6780 <br> Test: 0.7861 | Train: 0.6878 <br> Test: 0.5705 | 
| Most Important Features from Baseline + PROPHET + PageRank <br> (26 Features) | 2 min | Train: 0.6748 <br> Test: 0.5464 | Train: 0.6725 <br> Test: 0.5165 | Train: 0.6843 <br> Test: 0.7112 |
| All Baseline + PROPHET + PageRank <br> (36 Features) | 2 min | Train: 0.6804 <br> Test: 0.6089 | Train: 0.6785 <br> Test: 0.5942 | Train: 0.6884 <br> Test: 0.6758 |

**Feature Importance:** Consistently throughout the testing, the most influential features for logistic regression were precipitation, visiblity, wet bulb temperature, dry bulb temperature, and time between fligts.

**Cross-Validation Model Parameters Tested** For the logistic regression model the following parameters were tuned during cross-validation:
 - `regParam`: Regularization parameter, controls the strength of L2 regularization
 - `elasticNetParam`: Elastic net regularization parameter, a combination of L1 and L2 regularization
 - `threshold`: Threshold to make 0 or 1 prediction, where a threshold closer to 0 will make more 1 predictions and a threshold closer to 1 will make fewer 1 predictions.
 After cross-validation it was determined the best model parameters were `regParam` = 0.1, `elasticNetParam` = 0.0, and `threshold` = 0.4. For our desired metrics, it made sense a lower `threshold` resulted in better F-Beta and recall metrics.

###### Train and Test Results for delayed flights with 2015-2018 Cross-Validation Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:150px'> Wall Time </div>  | <div style='width:290px'> Weighted Avg F-Beta Score </div> | <div style='width:290px'> Weighted Avg Recall </div> | <div style='width:290px'> Weighted Avg Precision </div> | 
| ----- | ----- | ----- | ----- | ----- | ----- |
| Best Features from 1 Year <br> (36 Features) | `regParam` = 0.1, `elasticNetParam` = 0.0, `threshold` = 0.4 | 4 min | Train: 0.7977 <br> Test: 0.7977 | Train: 0.8716 <br> Test: 0.8567 | Train: 0.5957 <br> Test: 0.5975 | 


###### Train Results for delayed flights with 2015-2018 Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- | ----- |
| Best Model from XVal <br> (36 Features) | `regParam` = 0.1, `elasticNetParam` = 0.0, `threshold` = 0.4 | 4 min | Train: 0.7991 | Train: 0.8734 | Train: 0.5963 | 


#### Decision Tree
###### Train and Test Results for delayed flights with 2015 Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| Baseline Model <br> (31 Features; depth = 5) | 1 min | Train: 0.785 <br> Test: 0.464 | Train: 0.844 <br> Test: 0.510 | Train: 0.613 <br> Test: 0.343 | 
| Full Model + PROPHET + PageRank <br> (41 Features; depth = 5) | 1 min | Train: 0.781 <br> Test: 0.458 | Train: 0.837 <br> Test: 0.500 | Train: 0.616 <br> Test: 0.341 |
| Shortened Features + PageRank <br> (23 Features; depth = 5) | 1 min | Train: 0.396 <br> Test: 0.195 | Train: 0.361 <br> Test: 0.179 | Train: 0.648 <br> Test: 0.306 |
| Shortened Features + PageRank <br> (23 Features; depth = 10) | 1 min | Train: 0.782 <br> Test: 0.457 | Train: 0.839 <br> Test: 0.500 | Train: 0.616 <br> Test: 0.339 |

**Feature Importance:** Throughout feature selection, sinusoidal Scheduled Departure Time, Altimeter, Temperature, Precipitation, Time Between Flights, Tail Number and Origin showed importance. Tail Number and Origin were left out of the final model to reduce the bin size of the decision tree and overfitting on the high number of bins in those features.

**Cross-Validation Model Parameters Tested** For the logistic regression model the following parameters were tuned during cross-validation:
- `max_depth`: [6, 8, 10, 12, 14] Controls how the deep the decision tree can grow.
- `min_info_gain`: [1e-2, 1e-4, 1e-6] The minimum improvement gain required for a decision to split a node. 
- **Best Parameters**: The decision tree models were consistently overfitting during feature selection and cross validation. The best model parameters determined were `max_depth = 6` and `min_info_gain = 1e-2`, which had high performance yet contained a shallower and simpler tree structure to reduce overfitting.

###### Train and Test Results for delayed flights with 2015-2018 Cross-Validation Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> Weighted Avg F-Beta Score </div> | <div style='width:290px'> Weighted Avg Recall </div> | <div style='width:290px'> Weighted Avg Precision </div> | 
| ----- | ----- | ----- | ----- | ----- | ----- |
| Shortened Features + PageRank <br> (23 Features) | `max_depth = 6` <br> `min_info_gain = 1e-2`| 1 min | Train: 0.789 <br> Test: 0.550 | Train: 0.846 <br> Test: 0.596 | Train: 0.622 <br> Test: 0.422 | 


###### Train Results for delayed flights with 2015-2018 Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- | ----- |
| Shortened Features + PageRank <br> (23 Features) | `max_depth = 6` <br> `min_info_gain = 1e-2` | 1 min | Train: 0.789 | Train: 0.847 | Train: 0.619 | 



#### Random Forest
###### Train and Test Results for delayed flights with 2015 Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| Baseline Model <br> (19 Features, num_trees = 10, depth = 5) | 2 min | Train: 0.7536 <br> Test: 0.7728 | Train: 0.7836 <br> Test: 0.7946 | Train: 0.6535 <br> Test: 0.6965 |
| No Weather Features Model <br> (22 Features, num_trees = 10, depth = 10) | 2 min | Train: 0.7567 <br> Test: 0.7661 | Train: 0.7784 <br> Test: 0.7802 | Train: 0.6807 <br> Test: 0.7145 | 
| No Weather Features + No features with 0 importance <br> (15 Features, num_trees = 5, depth = 5) | 2 min | Train: 0.7537 <br> Test: 0.7704 | Train: 0.7852 <br> Test: 0.7926 | Train: 0.6494 <br> Test: 0.6930 |
| No Weather Features + No features with 0 importance <br> (15 Features, num_trees = 5, depth = 5) | 2 min | Train: 0.7537 <br> Test: 0.7704 | Train: 0.7852 <br> Test: 0.7926 | Train: 0.6494 <br> Test: 0.6930 |
| All Features <br> (36 Features, num_trees = 5, depth = 10) | 2 min | Train: 0.7642 <br> Test: 0.7710 | Train: 0.7884 <br> Test: 0.7856 | Train: 0.6807 <br> Test: 0.7174 |

**Feature Importance:** Throughout feature selection, all of the hourly based weather features were left out of the final feature set due to their low feature importance towards the classification of flight departures. The most important features encoded information on whether the previous flight was delayed.

**Cross-Validation Model Parameters Tested** For the Random Forest model the following parameters were tuned during cross-validation:
- `num_trees`: [5,10] Controls the number of trees in the ensemble.
- `depth`: [5,10] Controls how deep the decision trees in the ensemble can grow. 
- **Best Parameters**: The best model parameters determined were `num_trees = 10` and `depth = 5`, which generate the best performance containing a deeper tree structure and larger number of iterations.

###### Train and Test Results for delayed flights with 2015-2018 Cross-Validation Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> Weighted Avg F-Beta Score </div> | <div style='width:290px'> Weighted Avg Recall </div> | <div style='width:290px'> Weighted Avg Precision </div> | 
| ----- | ----- | ----- | ----- | ----- | ----- |
| Best Parameters Model (Baseline) <br> (19 Features) | `num_trees=10` <br> `depth=5` | 2 min | Train: 0.7633 <br> Test: 0.7681 | Train: 0.7974 <br> Test: 0.8044 | Train: 0.6524 <br> Test: 0.6529 | 


###### Train Results for delayed flights with 2015-2018 Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- | ----- |
| Best Parameters Model (Baseline) <br> (19 Features) | `num_trees=10` <br> `depth=5` | 2 min | Train: 0.7633 | Train: 0.7974 | Train: 0.6524 | 



#### Gradient Boosting
###### Train and Test Results for delayed flights with 2015 Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| Baseline Model <br> (19 Features, iteration=10, tree depth=5) | 1 min | Train: 0.742 <br> Test: 0.741 | Train: 0.758 <br> Test: 0.749 | Train: 0.683 <br> Test: 0.712 | 
| No weather features model <br> (22 Features, iteration=10, tree depth=10) | 1 min | Train: 0.768 <br> Test: 0.744 | Train: 0.786 <br> Test: 0.752 | Train: 0.706 <br> Test: 0.711 |
| No weather + No non-important features model <br> (12 Features, iteration=5, tree depth=5) | 1 min | Train: 0.739 <br> Test: 0.743 | Train: 0.755 <br> Test: 0.752 | Train: 0.679 <br> Test: 0.708 |
| Full Features Model <br> (36 Features, iteration=5, tree depth=10) | 1 min | Train: 0.768 <br> Test: 0.759 | Train: 0.787 <br> Test: 0.771 | Train: 0.702 <br> Test: 0.716 |

**Feature Importance:** The top 3 important features are the origin airport, delay of previous flight and time between flights.

**Cross-Validation Model Parameters Tested** For the Gradient Boosting model the following parameters were tuned during cross-validation:
- `max_iteration`: [5,10] Controls the maximum number of iterations the model can run.
- `max_tree_depth`: [5,10] Controls how the deep the decision tree can grow. 
- **Best Parameters**: The best model parameters determined were `max_iteration = 10` and `max_tree_depth = 10`, which generate the best performance.

###### Train and Test Results for delayed flights with 2015-2018 Cross-Validation Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> Weighted Avg F-Beta Score </div> | <div style='width:290px'> Weighted Avg Recall </div> | <div style='width:290px'> Weighted Avg Precision </div> | 
| ----- | ----- | ----- | ----- | ----- | ----- |
| Full Features Model <br> (36 Features) | `max_iterations=10` <br> `max_tree_depth=10` | 7 min | Train: 0.780 <br> Test: 0.755 | Train: 0.799 <br> Test: 0.779 | Train: 0.710 <br> Test: 0.675 | 


###### Train Results for delayed flights with 2015-2018 Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- | ----- |
| Full Features Model <br> (36 Features) | `max_iterations=10` <br> `max_tree_depth=10` | 7 min | Train: 0.742 | Train: 0.792 | Train: 0.698 | 


###### Early Stopping
Early stopping can save time and computer resources as it rules out unnecessary training which may result to worsen performance.  It also reduce the risk of over-fitting by decreasing the number of training iterations and lead to better performing model for testing.

We use the 2015 dataset as the validation set for early stoppping as it is not training and the held-out test set.  We got best iteration of 2 in the experiement. The metric we used is the ML metric of F-beta score but we want to avoid to stop too early, like after the first sign of poor performance as maybe the performance will get better after it get poor performance for one iteration.  So we introduce a patience parameter of 3 which allow the model to train for 3 addition loops after the last best validation score before stop and if the performance do get better during the 3 iterations, we will register the better performance and continue the training.  This helps to avoid stopping too early due to fluctuations in model performances.



#### MLP NN 
###### Train and Test Results for delayed flights with 2015 Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| Baseline Model <br> (31 Features) | 4 min | Train: 0.7011 <br> Test: 0.7002 | Train: 0.6731 <br> Test: 0.6821 | Train: 0.7001 <br> Test: 0.6902 | 
| 2-Layers Model <br> (23 Features) | 4 min | Train: 0.7000 <br> Test: 0.7011 | Train: 0.6861 <br> Test: 0.6733 | Train: 0.7110 <br> Test: 0.7035 |

**Feature Importance:** Origin airport, delay of previous flight and time between flights were found to be the most important features. 

**Cross-Validation Model Parameters Tested:** The 2-layer neural network (256 neurons per layer, ReLU activation, dropout rate 0.3, L2 regularization λ=0.01, learning rate 0.01) delivered the best weighted F1 score and validation loss. Adding more layers increased model complexity without significant performance gains, likely due to overfitting. Hyperparameter tuning helped optimize the architecture, yielding robust results for the flight delay prediction task.

###### Train and Test Results for delayed flights with 2015-2018 Cross-Validation Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> Weighted Avg F-Beta Score </div> | <div style='width:290px'> Weighted Avg Recall </div> | <div style='width:290px'> Weighted Avg Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| 2-Layers Model <br> (23 Features) | 12 min | Train: 0.7100 <br> Test: 0.7102 | Train: 0.6734 <br> Test: 0.7022 | Train: 0.7003 <br> Test: 0.7076 | 


###### Train Results for delayed flights with 2015-2018 Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:150px'> Wall Time </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| 2-Layers Model <br> (23 Features) | 14 min | Train: 0.7051 | Train: 0.7034 | Train: 0.7013 | 

**Early Stopping**

- Validation Accuracy: Measures the proportion of correctly classified instances.

- Validation Loss: Measures how well the model predicts compared to actual outcomes.

**Business Metrics:**

- Precision: If predicting delayed flights, focus on precision to minimize false positives (predicting delays that do not happen).

- Recall: If minimizing the number of missed delays, focus on recall.

- Cost Reduction: If each false positive or false negative has a business cost, the model should minimize total cost.

- Patience Parameter: Instead of quitting at the first sign of poor performance, training typically continues for a few more epochs or iterations. A patience parameter is often defined, which allows the model to train for a few additional epochs/trees to verify if the poor performance is temporary or a consistent trend.

**How Early Stopping Helps**

- Prevents Overfitting: Training on the validation set ensures the model does not memorize the training data and generalizes better to unseen data.

- Improves Generalization: Stops the model before it begins to overfit, resulting in better performance on the test set.

- Reduces Computational Cost: Saves time and resources by avoiding unnecessary epochs or trees, especially for large datasets or deep models.

- Ensures Robustness: By monitoring validation performance, early stopping selects a model with balanced performance, avoiding extremes in optimization.


#### Distributed Deep Learning Model 

**Feature Importance:** The same as the Neural Network, origin airport, delay of previous flight and time between flights were found to be the most important features.

**Cross-Validation Model Parameters Tested:** 3-layer architecture with two hidden layers, each containing 256 neurons with ReLU activation, L2 regularization (λ=0.01), and a dropout rate of 0.3. It used the Adam optimizer (learning rate = 0.01) and binary cross-entropy loss, trained for 25 epochs with a batch size of 64. Only one configuration was tested for the model, as it was a trial and much more research would be needed to expand upon this model.

###### Train and Test Results for delayed flights with 2015-2018 Cross-Validation Flight Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:150px'> Wall Time </div> |<div style='width:290px'> Weighted Avg F-Beta Score </div> | <div style='width:290px'> Weighted Avg Recall </div> | <div style='width:290px'> Weighted Avg Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| 3-Layers Model <br> (23 Features) | 18 min | Train: 0.7104 <br> Test: 0.7102 | Train: 0.6995 <br> Test: 0.6967 | Train: 0.6933 <br> Test: 0.6893 | 


## Discussion

Our results for each model are summarized below. We selected the Random Forest as our best model. The decision tree had the highest F-beta performance between test and train (0.789 vs. 0.795), but the model was consistently overfitting during training and cross-validation. Despite the slighly lower performance, the Random Forest should provide better generalizability on unknown data because they are less prone to overfitting when compared to decision trees. To verify, we validated our random forest model on both 2019 and 2020 flight data. The predictions for both years resulted in a similar F-beta scores, 0.768 for 2019 and 0.776 for 2020, verifying the generalizability of our model. The nineteen features in our random forest model includes:
- **Time Features:** quarter, scheduled departure time, month, day of week, days to the nearest holiday
- **Previous Flight Information:** planned time between flights, departure delay, departure delay group, departure time block, arrival delay, arrival delay group, arrival time block, flight route distance
- **Airport Characteristics:** origin type of previous flight
- **PageRank:** origin PageRank, destination PageRank

###### Summarized Train results for delayed flights with 2015-2018 Data and Test results for 2019 Data for Class = 1 (Delayed Flight)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:290px'>  F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| Logistic Regression <br> (36 Features) | `regParam = 0.1` <br> `elasticNetParam = 0.0` <br> `threshold = 0.4` | Train: 0.799 <br> Test: 0.679 | Train: 0.873 <br> Test: 0.677 | Train: 0.596 <br> Test: 0.689 | 
| Decision Tree <br> (23 Features) | `max_depth = 6` <br> `min_info_gain = 1e-2` | Train: 0.789 <br> Test: 0.795 | Train: 0.847 <br> Test: 0.861 | Train: 0.619 <br> Test: 0.609 | 
| Random Forest <br> (23 Features) | `max_depth = 5` <br> `num_trees = 10` | Train: 0.763 <br> Test: 0.768 | Train: 0.797 <br> Test: 0.804 | Train: 0.649 <br> Test: 0.653 | 
| Gradient Boosting <br> (23 Features) | `max_depth = 10` <br> `maxIter = 10` | Train: 0.742 <br> Test: 0.737 | Train: 0.792 <br> Test: 0.806 | Train: 0.698 <br> Test: 0.678 | 
| Neural Network <br> (23 Features) | `max_depth = 6` <br> `min_info_gain = 1e-2` | Train: 0.705 <br> Test: 0.703 | Train: 0.674 <br> Test: 0.707 | Train: 0.701 <br> Test: 0.710 | 

###### Confusion Matrix for our Random Forest model on 2019 data:
![Random Forest 2019 Confusion Matrix](https://github.com/jasondongmids/mids_w261_final_project/blob/main/ref/2019_confusion_matrix.png?raw=true)


###### Fianl Validation results for 2019 vs. 2020 Data for Class = 1 (Delayed Flights)
| <div style='width:150px'> Features </div> | <div style='width:290px'> Model Parameters </div> | <div style='width:290px'> F-Beta Score </div> | <div style='width:290px'> Recall </div> | <div style='width:290px'> Precision </div> | 
| ----- | ----- | ----- | ----- | ----- |
| Random Forest on 2019 <br> (23 Features) | `max_depth = 5` <br> `num_trees = 10` | Validation: 0.768 | Validation: 0.804 | Validation: 0.653 | 
| Random Forest on 2020 <br> (23 Features) | `max_depth = 5` <br> `num_trees = 10` | Validation: 0.776 | Validation: 0.846 | Validation: 0.831 | 

### Gap Analysis
The subsequent gap analysis is performed on our delayed predictions for the 2019 validation set. Our random forest model is performing well for departure delays in the morning, but performance slides throughout the day. We created a feature on the percentage of rolling delays per airport which could help the model capture this pattern later during the day. Our model has relatively stable predictions when looking at day of week and month, indicating that we've captured the seasonal patterns well on those time dimensions.

![Delay Predictions by Time Dimension](https://github.com/jasondongmids/mids_w261_final_project/blob/main/ref/delay_predictions_time.png?raw=true)

Surprisingly, our random forest did not include weather, airport, or airline features which could capture nuanced information for better predictions. Starting with airline companies, we see differences in recall which may be due to the sample imbalance between the companies. With changes to airline company landscape due to consolidation over time, we could further balance the dataset via stratified undersampling or keeping data continous by creating a custom index to map airline subsidiaries with one another such as with Envoy Airlines (MQ) and PSA Airlines (OH) with the American Airlines Group (AA) across time. Similar analysis could be done on the flight and tail numbers as airplanes are retired and routes are updated over time. Similar to airport delays, we can create a rolling delay feature by airline and airport to capture how companies are faring per airport.

![Delay Predictions by Airline Company](https://github.com/jasondongmids/mids_w261_final_project/blob/main/ref/delay_predictions_by_airline_company.png?raw=true)

We reviewed weather features - altimeter setting, precipitation, wind gust speed, and temperature - which were influential in the decision tree and logistic regression models. The percent of delayed predictions that were incorrect all exhibited similar behavior for wind gust speeds (below), thus did not see immediate patterns stand out. Since decision trees only use one feature per level to split, the logistic regression or neural network models may be better utilized to capture weather interactions. To keep the simplicity and generalizability of our random forest, we could further experiment with ensemble modeling for this complexity.

![Delay Predictions by Wind Gust Speed](https://github.com/jasondongmids/mids_w261_final_project/blob/main/ref/delay_predictions_by_wind_gust_speed.png?raw=true)

To improve our flight delay predictions in the future, we should consider adding features that capture more nuanced patterns and address existing gaps in our model. While our random forest model performs well for morning delays and captures seasonal patterns effectively, its performance declines later in the day. Additional temporal features could improve this. Incorporating weather-related variables (e.g., wind gust speed, precipitation, and temperature), which were influential in other models, could further enhance predictions. Further experimentation with ensemble models can strengthen our current model while mitigating weaknesses.

Although our current model trained on 2015-2018 data generalizes well on 2019 and 2020 data, we should update our model over time. Two options we would like to try include expanding our training window, 2015-2019, or to use a rolling window, 2016-2019, for refreshes. Testing both approachs can help us determine the optimal refresh strategy by balancing the utility of including more data while balancing the increase in time and resources for the larger training window.

## Conclusions and Next Steps 
To enable airline companies to mitigate costs from flight delays such as customer churn and refunds, we have created a random forest model to predict if a flight will be delayed (more than 15 minutes) two hours prior to its expected depature.

Our hypothesis was we can predict if a flight will be delayed based on weather and previous flight information. We have shown we can make these predictions with our random forest model and validation results. The journey to our successful results began with defining our primary evaluation metric - the F-beta score with \\(\beta = 2\\) to weigh the recall metric twice as heavily as precision. We believed that false negatives were more detrimental than false positives in that failing to identify a departure would risk breaking trust between the airline company and the customer.

To achieve the goal stated in our hypothesis, we had our baseline and model iterations revolve around logistic regression, decision trees, random forests, gradient boosted trees, neural networks, and a distributed deep learning model. We were able to show improvement from our initial logistic regression model (with balanced data and no feature engineering) with a train and test F-Beta of 0.625 and 0.699, respectively, on 2015 data. Our final model (with balanced data, feature engineering, 10 trees of depth 5) improved to an F-Beta score of 0.768 for 2019 validation data and 0.776 for 2020 data.

The selected features within our final model were time features (quarter, scheduled departure time, month, day of week, days to the nearest holiday), previous flight information (planned time between flights, departure delay, departure delay group, departure time block, arrival delay, arrival delay group, arrival time block, flight route distance), origin airport type classification, and origin and destination airport PageRank. The most influential features in our final model were based on whether the previous flight was delayed, illustrating the importance of past delay information in predicting future flight delays. Moreover, the superior performance of our random forest model as witnessed in our validation steps demonstrates the power of ensemble learning in balancing the predictive power of our custom feature engineering with generalizbility to unseen data.

Based on our gap analysis we showed our model was consistent with the day of the week and month but was inconsistent on the time of day (performing best at 5am and worst and 10pm). It was also inconsistent across airlines. As the random forest model did not include any weather features in the resulting model, if we were to continue this project we would explore an ensemble model that included our logistic regression and decision tree models (both containing weather features) with the random forest model. We believe, this would capture some of these missed predictions.

If continuing to work on this project we would like to expand our data to plane and mechanical history next. If we had plane flight information such as total hours flow, hours since last routine maintenance, and number of unexpected maintenance events we could further improve our model and user experience.

# Team Members
- Jason Dong
- Nick Gasser
- Sameer Karim
- Anson Quon
- Gilbert Wong

