# %%
# 1. Importing necessary modules and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import random

# %%
base_dir = "MHealth Dataset\data\mHealth_subject"

# %%
def d_types_report(df):
    columns=[]
    d_types=[]
    uniques=[]
    n_uniques=[]
    null_values=[]
    null_values_percentage=[]

    for i in df.columns:
        columns.append(i)
        d_types.append(df[i].dtypes)
        uniques.append(df[i].unique()[:5])
        n_uniques.append(df[i].nunique())
        null_values.append(df[i].isna().sum())
        null_values_percentage.append(null_values[-1] * 100 / df.shape[0])

    return pd.DataFrame({"Columns": columns, "Data_Types": d_types, "Unique_values": uniques, "N_Uniques": n_uniques,  "Null_Values": null_values, "Null_Values_percentage": null_values_percentage})

def randomColor():
    r = lambda: random.randint(0,255)
    return ('#%02X%02X%02X' % (r(),r(),r()))
randomColor()

def get_variable_name(variable):
    globals_dict = globals()
    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable]

# %%
variables = locals()
dfs = []
for i in range(1, 11):
    globals()['df%s'%str(i)] = pd.read_csv(base_dir + str(i) + '.csv')
    dfs.append(variables.get('df'+str(i)))

df = pd.concat(dfs)
df = df.dropna(how='any',axis=0)
d_types_report(df)

# %%
plt.rcParams["figure.figsize"] = (20,4)
for df_ in dfs:
    color=randomColor()
    for column_name in df_.columns:
        if column_name == 'Label':
            continue
        plt.figure()
        plt.plot(df_[column_name], color=color)
        plt.title(column_name)
        plt.show

# %%
# Handle class imbalance

from sklearn.utils import resample

df_majority = df[df.Label==0]
df_minorities = df[df.Label!=0]

df_majority_downsampled = resample(df_majority,n_samples=300000)
df = pd.concat([df_majority_downsampled, df_minorities])


df_majority = df[df.Label != 12.0]
df_minorities = df[df.Label ==12]
df_minority_upsampled = resample(df_minorities,n_samples=30720)
df = pd.concat([df_minority_upsampled, df_majority])

df.Label.value_counts()

# %%
# 3. Preprocess the data
# Encode activity labels
activity_labels = {activity: idx for idx, activity in enumerate(df['Label'].unique())}
df['ActivityEncoded'] = df['Label'].apply(lambda x: activity_labels[x])

# %%
# Split the data into features and labels
X = df.drop(['Label', 'ActivityEncoded'], axis=1).values
y = df['ActivityEncoded'].values

# %%
# Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# 4. Use LSTM Architecture - both baseline and multi-channel
# Baseline LSTM

baseline_lstm = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(len(activity_labels), activation='softmax')
])

# %%
# Multi-channel LSTM

multi_channel_lstm = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dropout(0.2),
    BatchNormalization(),
    Dense(len(activity_labels), activation='softmax')
])

# %%
def aggregate_weights(local_models):
    # Initialize global_weights as None to handle the first model's weights
    global_weights = None
    for model in local_models:
        # Get the weights of the current model
        model_weights = model.get_weights()
        
        # If global_weights is None, set it to the current model's weights
        if global_weights is None:
            global_weights = model_weights
        else:
            # Otherwise, add the current model's weights to the global_weights
            for i in range(len(global_weights)):
                global_weights[i] += model_weights[i]
    
    # Average the global_weights by dividing by the number of local models
    global_weights = [w / len(local_models) for w in global_weights]
    
    return global_weights

# %%
# 6. Train and evaluate the models
from keras.layers import Input
from keras.models import Model

num_clients = 10
local_models = []

for _ in range(num_clients):
    # Create a local dataset for the client
    local_X_train, local_y_train = X_train[np.random.choice(len(X_train), size=len(X_train) // num_clients)], \
                                   y_train[np.random.choice(len(y_train), size=len(y_train) // num_clients)]

    # Define input shape
    input_shape = (X_train.shape[1], 1)
    inputs = Input(shape=input_shape)

    # Build LSTM layers
    x = LSTM(128, return_sequences=True)(inputs)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    outputs = Dense(len(activity_labels), activation='softmax')(x)

    # Create model
    local_model = Model(inputs=inputs, outputs=outputs)
    local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    local_model.fit(local_X_train, local_y_train, epochs=10, batch_size=32, verbose=0)
    local_models.append(local_model)

    # Print local model performance
    local_preds = local_model.predict(X_test)
    local_accuracy = accuracy_score(y_test, np.argmax(local_preds, axis=1))
    print(f"Local Model {len(local_models)} Accuracy: {local_accuracy:.2f}")

  # Aggregate local models using FedAvg
global_weights = aggregate_weights(local_models)

# # Perform additional training using FedSGD
# for epoch in range(1):
#     global_model = federated_sgd(local_models, global_model)
#     global_preds = global_model.predict(X_test)
#     global_accuracy = accuracy_score(y_test, np.argmax(global_preds, axis=1))
#     print(f"Epoch {epoch + 1}: Global Accuracy: {global_accuracy:.2f}")

# %%

# Initialize the global model with the aggregated weights
global_model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dropout(0.2),
    BatchNormalization(),
    Dense(len(activity_labels), activation='softmax')
])
global_model.set_weights(global_weights)

# Compile the global model
global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the global model on the entire dataset
avg = global_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Save the global model
global_model.save('global_Avg_model.h5')

# %%
from keras.models import load_model
from sklearn.metrics import classification_report

# # Assuming `global_model` is your loaded model and `X_test` and `y_test` are your test data
# global_model.evaluate(X_test, y_test, verbose=0)

# Load the global model
global_model = load_model('global_Avg_model.h5')

# Assuming global_model is already trained and global_preds are predictions on the test set
global_preds = global_model.predict(X_test)

# Convert predictions to class labels
global_pred_labels = np.argmax(global_preds, axis=1)


# Calculate accuracy
FedAvg_accuracy = accuracy_score(y_test, global_pred_labels)
print(f"Global Model Accuracy: {FedAvg_accuracy:.2f}")


# Generate a classification report
FedAvg_report = classification_report(y_test, global_pred_labels, output_dict=True)
print(FedAvg_report)


# %%
import matplotlib.pyplot as plt
import numpy as np

# Extract the F1-score, recall, and precision values for each class
f1_score_values = [report['f1-score'] for key, report in FedAvg_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]
recall_values = [report['recall'] for key, report in FedAvg_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]
precision_values = [report['precision'] for key, report in FedAvg_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]

# Create labels for the x-axis
labels = [key for key in FedAvg_report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot F1-score values for each class
ax.plot(labels, f1_score_values, marker='o', label='F1-score')

# Plot recall values for each class
ax.plot(labels, recall_values, marker='o', label='Recall')

# Plot precision values for each class
ax.plot(labels, precision_values, marker='o', label='Precision')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('F1-score, Recall, and Precision by Class')
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels) # Rotate labels for better readability
ax.legend()

fig.tight_layout()

plt.show()

# %%
from sklearn.metrics import confusion_matrix

# Assuming y_test are the true labels and global_pred_labels are the predicted labels
cm = confusion_matrix(y_test, global_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %%
def federated_sgd(local_models, global_lstm_model, num_rounds=10):
    global_weights = global_lstm_model.get_weights()
    
    for _ in range(num_rounds):
        local_updates = []
        for local_model in local_models:
            local_weights = local_model.get_weights()
            local_updates.append(local_weights)
        
        # Aggregate updates
        global_weights = [np.mean(np.array(updates), axis=0) for updates in zip(*local_updates)]
        
        # Update global model
        global_lstm_model.set_weights(global_weights)
    
    return global_lstm_model

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
# Define the global model architecture
global_sgd_model = Sequential([
    LSTM(256, input_shape=(X_train.shape[1], 1), return_sequences=True),
    LSTM(256, return_sequences=True),
    LSTM(128),
    Dropout(0.3),
    Dense(len(activity_labels), activation='softmax')
])

# %%
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import SGD

# Define the local model architecture for Federated Learning
def create_local_sgd_model():
    local_sgd_model = Sequential([
        LSTM(256, input_shape=(X_train.shape[1], 1), return_sequences=True),
        Dropout(0.4),
        LSTM(256, return_sequences=True),
        Dropout(0.4),
        LSTM(128),
        Dropout(0.4),
        
        Dense(len(activity_labels), activation='softmax')
    ])
    return local_sgd_model

# Initialize local models for Federated Learning
num_clients_sgd = 10 # Number of local models (clients) for Federated Learning
local_sgd_models = [create_local_sgd_model() for _ in range(num_clients_sgd)]

# Train local models with increased LSTM units for Federated Learning
local_sgd_preds_series = [] # Initialize an empty list to store predictions
for local_sgd_model in local_sgd_models:
    local_sgd_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    local_sgd_model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    # Get predictions from the local model
    local_sgd_preds = local_sgd_model.predict(X_test)
    local_sgd_preds_series.append(local_sgd_preds) # Append predictions to the list

# Concatenate all local model predictions into a single series
local_sgd_preds_series = np.concatenate(local_sgd_preds_series, axis=0)

# Print local model performance for Federated Learning
for i, local_sgd_model in enumerate(local_sgd_models):
    local_sgd_preds = local_sgd_preds_series[i]
    local_sgd_accuracy = accuracy_score(y_test, np.argmax(local_sgd_preds, axis=1))
    print(f"Local Federated Model {i+1} Accuracy: {local_sgd_accuracy:.2f}")

# Perform federated learning for a specified number of rounds


# %%
global_fed_sgd_model = federated_sgd(local_sgd_models, global_sgd_model, num_rounds=10)

# %%
from keras.callbacks import EarlyStopping

# Compile the global model with SGD optimizer
sgd_optimizer = SGD(learning_rate=0.001, momentum=0.9) # Adjust learning rate

global_fed_sgd_model.compile(optimizer=sgd_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Add dropout layers for regularization
global_fed_sgd_model.add(Dropout(0.5))

# Train the global model using SGD with early stopping
sgd = global_fed_sgd_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping])

# Save the global model
global_fed_sgd_model.save('global_sgd_model.keras')

# %%
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the global model
global_fed_sgd_model = load_model('global_lstm_model.h5')


# Assuming `global_lstm_model` is your trained model and `X_test` is your test data
global_preds = global_fed_sgd_model.predict(X_test)

# Convert predictions to class labels
global_pred_labels = np.argmax(global_preds, axis=1)

# Calculate accuracy
FedSGD_accuracy = accuracy_score(y_test, global_pred_labels)
print(f"Fed SGD Model Accuracy: {FedSGD_accuracy:.2f}")

# Generate a classification report
FedSGD_report = classification_report(y_test, global_pred_labels, output_dict=True)
print(FedSGD_report)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Extract the F1-score, recall, and precision values for each class
f1_score_values = [report['f1-score'] for key, report in FedSGD_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]
recall_values = [report['recall'] for key, report in FedSGD_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]
precision_values = [report['precision'] for key, report in FedSGD_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]

# Create labels for the x-axis
labels = [key for key in FedSGD_report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot F1-score values for each class
ax.plot(labels, f1_score_values, marker='o', label='F1-score')

# Plot recall values for each class
ax.plot(labels, recall_values, marker='o', label='Recall')

# Plot precision values for each class
ax.plot(labels, precision_values, marker='o', label='Precision')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('F1-score, Recall, and Precision by Class')
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels) # Rotate labels for better readability
ax.legend()

fig.tight_layout()

plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Assuming FedAvg_report and FedSGD_report are dictionaries with the same structure
# Extract the precision, recall, and F1-score values for each class
fed_avg_precision = [report['precision'] for key, report in FedAvg_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]
fed_avg_recall = [report['recall'] for key, report in FedAvg_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]
fed_avg_f1_score = [report['f1-score'] for key, report in FedAvg_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]

fed_sgd_precision = [report['precision'] for key, report in FedSGD_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]
fed_sgd_recall = [report['recall'] for key, report in FedSGD_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]
fed_sgd_f1_score = [report['f1-score'] for key, report in FedSGD_report.items() if key not in ['accuracy', 'macro avg', 'weighted avg']]

# Create a figure and axis for precision vs recall
fig, ax = plt.subplots(figsize=(12, 5))

# Plot precision vs recall for FedAvg_report
ax.scatter(fed_avg_precision, fed_avg_recall, marker='o', label='FedAvg Precision vs Recall')

# Plot precision vs recall for FedSGD_report
ax.scatter(fed_sgd_precision, fed_sgd_recall, marker='o', label='FedSGD Precision vs Recall')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precision vs Recall for FedAvg and FedSGD Reports')
ax.legend()

fig.tight_layout()

plt.show()

# Create a new figure and axis for F1-score vs recall
fig, ax = plt.subplots(figsize=(12, 5))

# Plot F1-score vs recall for FedAvg_report
ax.scatter(fed_avg_f1_score, fed_avg_recall, marker='o', label='FedAvg F1-score vs Recall')

# Plot F1-score vs recall for FedSGD_report
ax.scatter(fed_sgd_f1_score, fed_sgd_recall, marker='o', label='FedSGD F1-score vs Recall')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('F1-score')
ax.set_ylabel('Recall')
ax.set_title('F1-score vs Recall for FedAvg and FedSGD Reports')
ax.legend()

fig.tight_layout()

plt.show()

# Create a new figure and axis for F1-score vs recall
fig, ax = plt.subplots(figsize=(12, 5))

# Plot F1-score vs recall for FedAvg_report
ax.scatter(fed_avg_precision, fed_avg_f1_score, marker='o', label='FedAvg Precision vs F1-score')

# Plot F1-score vs recall for FedSGD_report
ax.scatter(fed_sgd_precision, fed_sgd_f1_score, marker='o', label='FedSGD Precision vs F1-score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('F1-score')
ax.set_ylabel('Recall')
ax.set_title('F1-score vs Recall for FedAvg and FedSGD Reports')
ax.legend()

fig.tight_layout()

plt.show()


