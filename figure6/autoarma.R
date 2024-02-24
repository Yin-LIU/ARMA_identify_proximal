library(forecast)

stock <- read.csv("Netflix_stock_history.csv")
stock.data <- matrix((stock$Close))

stock.train <- stock.data[1:365]
stock.test <- stock.data[366]

plot(stock.train)

ARMAfit <- auto.arima(stock.train, max.p = 5, max.q = 5,stepwise=FALSE, max.order=10,seasonal = FALSE)



forecast(ARMAfit,h=10)
