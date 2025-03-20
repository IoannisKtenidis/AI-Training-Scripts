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

# Label encode the three categories in the reservation_status column
hotels['reservation_status'] = hotels['reservation_status'].replace({'Check-Out':2, 'Canceled':1, 'No-Show':0})

# Create Training and Testing Sets
# Create a list named train_features that contains all of the feature names (column names excluding the target variables is_canceled and reservation_status).
# Remove target columns
remove_cols = ['is_canceled', 'reservation_status']
# Select training features
train_features = [x for x in hotels.columns if x not in remove_cols]

#Create the X and y tensors
X = torch.tensor(hotels[train_features].values, dtype=torch.float)
y = torch.tensor(hotels['reservation_status'].values, dtype=torch.long)

# Split the X and y tensors into training and testing splits
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    test_size=0.2, 
                                                    random_state=42) 
print("Training Shape: ", X_train.shape)
print("Testing Shape: ", X_test.shape)

# Architecture
torch.manual_seed(42)
​
multiclass_model = nn.Sequential(
    nn.Linear(65,65),
    nn.ReLU(),
    nn.Linear(65,36),
    nn.ReLU(),
    nn.Linear(36,3)
)
# Loss FUnction
loss == nn.CrossEntropyLoss()
optimizer = optim.Adam(multiclass_model.parameters(), lr=0.1)

#Build the training loop to train our neural network.

num_epochs = 500
for epoch in range(num_epochs):
    predictions = multiclass_model(X_train)
    CELoss = loss(predictions, y_train)
    CELoss.backward()
    optimizer.step()
    optimizer.zero_grad()
​
    if (epoch + 1) % 100 == 0:
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], CELoss: {CELoss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# Evaluate the model
multiclass_model.eval()
with torch.no_grad():
    multiclass_predictions = multiclass_model(X_test)
    multiclass_predicted_labels = torch.argmax(multiclass_predictions, dim=1)

# accuracy, precision, recall, and F1 score

multiclass_accuracy = accuracy_score(y_test, multiclass_predicted_labels)
print(f'Accuracy: {multiclass_accuracy.item():.4f}')

multiclass_report = classification_report(y_test, multiclass_predicted_labels)
print("Classification Report:\n", multiclass_report)
