# Customer-Lifetime-Value-Prediction-Online-Retail-Dataset

CLTV is a method for predicting a how much a customer is going to create value for a firm in a specific time. 

CLTV method consists of two main parts: 
  1. BG/NGD (Beta Geometric/ Negative Binomial Distributions) Submodel: Calculation of conditional expected number of transactions,
  2. Gamma Gamma Submodel: Calculation of conditional expected average profit.

BetaGeoFitter is used for CLTV model. 

While the predict functions' of submodels are timed for a week, BeteGeoFitter functions time variable is calculated monthly.

Online Retail II Dataset Information:
      (https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

Dataset inludes transactions non-store online retail between 01/12/2009 and 09/12/2011.
The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.

Dataset variables:

InvoiceNo: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.
StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
Description: Product (item) name. Nominal.
Quantity: The quantities of each product (item) per transaction. Numeric.
InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated.
UnitPrice: Unit price. Numeric. Product price per unit in sterling (Â£).
CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
Country: Country name. Nominal. The name of the country where a customer resides.



For better resoruces about the topic, visit: (https://www.veribilimiokulu.com/)
