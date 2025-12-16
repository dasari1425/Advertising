# Advertising
This regression model will allow inference to determine the best advertising approach for generating sales.

An Advertising.csv file is available. 
This dataset contains data about the sales of a product in relation to the advertising budgets spent on TV, radio and newspaper. It's commonly used to explore the relationship between advertising efforts and sales.  There are 200 samples each with 4 attributes.

train.py will train the regression model and produce a model.pkl file.

A sample index.html page is available which will accept inputs in the form of features to generate output/inference from the model.pkl.


Lastly, the app.py will serve the model. It will load the pickel file and index.html file, submit features from the form/index.html, and return predictions from the model based on the features entered.
