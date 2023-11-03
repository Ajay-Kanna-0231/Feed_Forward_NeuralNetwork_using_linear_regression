import pandas as pd
import numpy as np

# Read the data for cats and dogs into separate dataframes 
# These are arrays which contain pixel values of 80 images, 1 image corresponds to 1024*1)
cat_data = pd.read_csv(r'C:\Users\Ajay Kanna\Desktop\iitm\sem3\AS2101\assgn9\catData.txt', sep=",", header=None)
dog_data = pd.read_csv(r'C:\Users\Ajay Kanna\Desktop\iitm\sem3\AS2101\assgn9\dogData.txt', sep=",", header=None)  

print(cat_data.shape)
print(dog_data.shape)

# Concatenate the data to create a feature matrix 3
# here we are taking the 60 images of each cat and dog for training
feature_matrix = pd.concat([cat_data.iloc[:, :60], dog_data.iloc[:, :60]], axis=1)

# Transpose the feature matrix for proper shape
feature_matrix = feature_matrix.T

# Create the target vector (1 for cats, 0 for dogs)
target_vector = pd.Series([1] * 60 + [0] * 60)

# Calculate the pseudo-inverse of the feature matrix
feature_matrix_inverse = np.linalg.pinv(feature_matrix)

# Calculate the weights for the linear regression model
weights = np.dot(feature_matrix_inverse, target_vector.values)
print("Weights:", weights)

# Prepare the test feature matrix for the remaining 20 cat and 20 dog images
test_feature_matrix = pd.concat([cat_data.iloc[:, 60:80], dog_data.iloc[:, 60:80]], axis=1)
test_feature_matrix = test_feature_matrix.T

# Predict class labels for test data
predictions = np.dot(test_feature_matrix, weights)

# Define the true class labels for the test data
true_labels = [1] * 20 + [0] * 20

# Threshold the predictions to classify as cats (1) or dogs (0)
predicted_labels = [1 if prediction >= 0.5 else 0 for prediction in predictions]

# Calculate and print the accuracy of the model
accuracy = sum(1 for i in range(40) if predicted_labels[i] == true_labels[i]) / 40.0
print("Accuracy:", accuracy)