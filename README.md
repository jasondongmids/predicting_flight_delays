# Predicting Flight Delays to Mitigate Potential Delays and Costs

Read the full report here: [Predicting Flight Delays.md](https://github.com/jasondongmids/predicting_flight_delays/blob/main/Predicting%20Flight%20Delays.md)

## Project Abstract

Airline On-Time Performance, defined as a flight arriving within 15 minutes of its scheduled arrival time, is a critical metric for airlines and regulators. Analysis of TranStats flight data from the US Department of Transportation (USDOT) reveals that 18% of flights were delayed by more than 15 minutes between 2015 and 2021. Delays incur substantial operational costs, especially with the USDOT’s ruling requiring automatic refunds for domestic flights delayed by three hours or more. To address this, our project focuses on predicting flight delays two hours before departure, with a primary goal of minimizing false negatives to preserve trust and prevent customer churn.

Our pipeline incorporates flight data from TranStats and weather data from NOAA to build a robust machine learning model, excluding delay causes like mechanical issues or staffing shortages to maintain practical generalizability. Baseline logistic regression models achieved an F-Beta test score of 0.699 with 21 features, improving to 0.714 when 6 custom features—such as previous flight delays and cyclical time—were added. Experiments with random forests, bagging, boosting, and neural networks identified random forests as the best-performing model, achieving superior accuracy and an F-Beta score of 0.765 on the test set.

Our random forest model demonstrated the best ability to capture complex relationships in the data while maintaining interpretability, making them ideal for predicting delays in real-world scenarios. This model resulted in a 0.768 F-beta score on unseen 2019 test data. Our model trained on 2015-2018 also showed strong generalizability when testing on 2020 data with a F-beta score of 0.776. While our initial model shows strong promise in predicting delays, we see opportunities of improvement by incorporating additional features and trying ensembling to further optimize the model for operational use across the airline industry.
