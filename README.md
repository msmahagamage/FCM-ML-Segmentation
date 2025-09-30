# Hybrid Fuzzy C-Means and Maximum Likelihood Classifier for Image Segmentation

This MATLAB project provides a robust, vectorized implementation of a hybrid classification workflow that uses Fuzzy C-Means (FCM) to initialize a Maximum Likelihood Classifier (MLC) for color image segmentation.

This method combines the unsupervised clustering power of FCM to automatically determine class statistics with the statistical rigor of the MLC for a precise final classification.



---

##  Features

- **Hybrid Approach:** Leverages the strengths of both Fuzzy C-Means and Maximum Likelihood classification.
- **Modular and Reusable:** The core FCM algorithm is encapsulated in a well-documented function (`fcmcluster.m`) that can be used in other projects.
- **Flexible:** Easily configure the number of classes and key algorithm parameters in the main driver script.
- **Clear Visualization:** The final output is a color-coded segmented image with a legend, making the results easy to interpret.

---

##  How It Works

The classification process follows these steps:

1.  **Image Preparation:** The input color image is loaded and converted into a `N x 3` matrix of pixel data, where N is the total number of pixels.
2.  **Fuzzy C-Means Initialization:** The `fcmcluster.m` function is run on the pixel data. Its purpose is not to produce the final classification, but to find the statistical centers (mean color vectors) of the desired number of classes in an unsupervised manner.
3.  **Fuzzy Covariance Calculation:** Using the membership values from the FCM result, the script calculates a fuzzy covariance matrix for each class. This matrix describes how the colors within a class vary.
4.  **Maximum Likelihood Classification:** Each pixel is then formally classified using the mean vectors and covariance matrices derived from the fuzzy step. The algorithm calculates the statistical likelihood that a pixel belongs to each class and assigns it to the class with the highest probability.
5.  **Visualization:** The resulting class labels are reshaped back into an image and displayed with a color map for easy analysis.

---

##  Requirements

- MATLAB (R2020a or newer recommended)
- Image Processing Toolbox (for `imread`, `imshow`, `label2rgb`)
- Statistics and Machine Learning Toolbox (for `pdist2`)

---

##  How to Use

1.  Clone this repository to your local machine.
2.  Ensure the `fcmcluster.m` function and your main script (e.g., `runHybridClassification.m`) are in the same directory or that `fcmcluster.m` is in your MATLAB path.
3.  Open the main script (`runHybridClassification.m`).
4.  Change the `imagePath` variable to point to your own image file.
5.  Adjust the `numClasses` variable and any optional parameters as needed.
6.  Run the script.

### Example Driver Script (`runHybridClassification.m`)

```matlab
% --- 1. Load and Prepare Image Data ---
imagePath = 'C:\my\fcm\Subset\subset.tiff'; % <-- CHANGE THIS TO YOUR IMAGE
originalImage = imread(imagePath);
imageData = im2double(originalImage);
[rows, cols, numBands] = size(imageData);
pixelData = reshape(imageData, rows * cols, numBands);

% --- 2. Run FCM to Find Class Statistics ---
numClasses = 3; % For example: Vegetation, Building, Soil
fcmOptions.fuzziness = 2.0;
[fuzzyMeans, U] = fcmcluster(pixelData, numClasses, fcmOptions);

% --- 3. Calculate Fuzzy Covariances ---
% ... (Full logic in the script) ...

% --- 4. Perform Maximum Likelihood Classification ---
% ... (Full logic in the script) ...

% --- 5. Visualize and Save the Final Result ---
% ... (Full logic in the script) ...
```

---

