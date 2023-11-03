# Feed_Forward_NeuralNetwork_using_linear_regression

# Linear Regression Classifier for Cat and Dog Images

This Python code demonstrates a simple linear regression model for classifying images of cats and dogs based on pixel values. The code uses the Pandas and NumPy libraries for data manipulation and linear algebra operations. This code will be training 60 images of dog and cat each to find the weights and using those weights we test 20 images of cat and dog each and will find the test accuracy!

## Data

The code assumes the existence of two text files containing pixel values for images of cats and dogs. The data is organized such that each image is represented as a single column with pixel values in columns.

- `catData.txt`: Contains pixel values for cat images.
- `dogData.txt`: Contains pixel values for dog images.

## How to Use

1. Ensure you have the required data files (`catData.txt` and `dogData.txt`) in your specified file path.
2. Install the necessary Python libraries, Pandas and NumPy, if not already installed.

    ```bash
    pip install pandas numpy
    ```

3. Open the Python script and replace the file paths in the code with the actual paths to your data files.

4. Run the script. It will perform the following steps:

   - Read the data from the text files into Pandas DataFrames.
   - Concatenate the data to create a feature matrix for training.
   - Transpose the feature matrix for proper shape.
   - Create a target vector (1 for cats, 0 for dogs).
   - Calculate the pseudo-inverse of the feature matrix.
   - Calculate the weights for the linear regression model.
   - Prepare the test feature matrix for the remaining images.
   - Predict class labels for test data.
   - Calculate and print the accuracy of the model.

## Results

The code calculates and prints the accuracy of the linear regression model in classifying cat and dog images. The accuracy represents the proportion of correctly classified images.

## Note

This code is a simplified example of image classification using linear regression. In practice, more advanced machine learning models and image processing techniques would be used for accurate image classification.

Feel free to modify the code and data to suit your specific requirements and experiment with different machine learning algorithms for image classification.
