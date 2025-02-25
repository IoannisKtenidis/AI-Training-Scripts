# Use PyTorch to Predict Cancellations of reservations
# Your goal in this project is build and train a neural network to predict if a customer will cancel their hotel booking reservation based on data including the booking dates, average daily cost, number of adults/children/babies, duration of stay, and so forth.
# Set up libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Import the CSV FILE to a pandas DataFrame
hotels = pd.read_csv("FILE.csv")
# Preview the first five rows
hotels.head()
#Use the .info() method on the hotels DataFrame to inspect the data.
hotels.info() 

# the .value_counts() method on the is_canceled column to count the number and the percentage of overall cancellations.
# Number of cancellations
print(hotels['is_canceled'].value_counts(0))
​
# Percentage of cancellations
print(hotels['is_canceled'].value_counts(1))

# The reservation_status column tells us if the booking was canceled while also telling us if the customer was a no-show.
# Number of cancellations
print(hotels['reservation_status'].value_counts(0))
​
# Percentage of cancellations
print(hotels['reservation_status'].value_counts(1))

# .groupby() method to group the data by the arrival_date_month column and apply the .mean() aggregation function on the is_canceled column. This will return the percent of reservations cancelled in each month.
# .sort_values() method to sort the percentages from lowest to highest.

cancellations_by_month = hotels.groupby('arrival_date_month')['is_canceled'].mean()
cancellations_by_month.sort_values()

# Drop any columns you don't want to use to train a cancellation model
object_columns = ['arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']
hotels[object_columns].head()
drop_columns = ['country', 'agent', 'company', 'reservation_status_date',
                'arrival_date_week_number', 'arrival_date_day_of_month', 'arrival_date_year']
​hotels = hotels.drop(labels=drop_columns, axis=1)

# encode the column which tells us which type of meal(s) the customer booked:
hotels['meal'] = hotels['meal'].replace({'Undefined':0, 'SC':0, 'BB':1, 'HB':2, 'FB':3})

# prepare the rest of the categorical columns using one-hot encoding.
one_hot_columns = ['arrival_date_month', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'market_segment']
hotels = pd.get_dummies(hotels, columns=one_hot_columns, dtype=int)
hotels.head()

# Create Training and Testing Sets
# Create a list named train_features that contains all of the feature names (column names excluding the target variables is_canceled and reservation_status).
# Remove target columns
remove_cols = ['is_canceled', 'reservation_status']
# Select training features
train_features = [x for x in hotels.columns if x not in remove_cols]
X = torch.tensor(hotels[train_features].values, dtype=torch.float)
y = torch.tensor(hotels['is_canceled'].values, dtype=torch.float).view(-1,1)

# split data contained in X and y into training and testing sets.
from sklearn.model_selection import train_test_split
​
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80,
                                                    test_size=0.20,
                                                    random_state=42) 
​
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

#Architecture
torch.manual_seed(42)
​
model = nn.Sequential(
    nn.Linear(65, 36),
    nn.ReLU(),
    nn.Linear(36, 18),
    nn.ReLU(),
    nn.Linear(18, 1),
    nn.Sigmoid()
)

#loss function and optimizer used for training:
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# training loop to train neural network.
from sklearn.metrics import accuracy_score
​
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    BCELoss = loss(predictions, y_train)
    BCELoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions >= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# evaluate the trained neural network on the testing set:
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_predicted_labels = (test_predictions >= 0.5).int()
 
# accuracy, precision, recall, and F1 scores using the sklearn.metrics module:​
test_accuracy = accuracy_score(y_test, test_predicted_labels)
print(f'Accuracy: {test_accuracy.item():.4f}')
report = classification_report(y_test, test_predicted_labels)
print("Classification Report:\n", report)