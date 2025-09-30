clear; clc; close all;

% --- 1. Load and Prepare Image Data ---
try
    originalImage = imread('C:\my\fcm\Subset\subset.tiff');
catch
    error('Could not find the image file. Please check the path.');
end

% Convert image to a numeric type (0-1 range) for stable calculations
imageData = im2double(originalImage);

% Reshape the (rows x cols x 3) image into a (numPixels x 3) matrix
% This is the standard, professional way to prepare image data for clustering.
[rows, cols, numBands] = size(imageData);
pixelData = reshape(imageData, rows * cols, numBands);

% --- 2. Use FCM to Derive Class Statistics ---
% The goal of this step is to find the mean color and covariance for
% each class in an unsupervised way.
numClasses = 3; % Vegetation, Building, Soil
fcmOptions.fuzziness = 2.0;
fcmOptions.maxIter = 50;

fprintf('Running Fuzzy C-Means to find initial class statistics...\n');
tic;
% Use the reusable function to get fuzzy clusters
[fuzzyMeans, U] = fcmcluster(pixelData, numClasses, fcmOptions);
toc;

% --- 3. Calculate Fuzzy Covariance for Each Class ---
% These stats will be used by the Maximum Likelihood Classifier
fprintf('Calculating fuzzy covariance matrices...\n');
fuzzyCovariances = zeros(numBands, numBands, numClasses);
mf = U.^fcmOptions.fuzziness;

for k = 1:numClasses
    % Get the membership weights for this class
    classWeights = mf(k, :)';
    
    % Subtract the mean from the data
    dataMinusMean = pixelData - fuzzyMeans(k, :);
    
    % Calculate the fuzzy covariance matrix (vectorized)
    numerator = (dataMinusMean' .* classWeights') * dataMinusMean;
    denominator = sum(classWeights);
    fuzzyCovariances(:,:,k) = numerator / denominator;
end

% --- 4. Perform Maximum Likelihood Classification ---
% This is the final classification step, using the stats from FCM.
fprintf('Performing Maximum Likelihood Classification...\n');
tic;
logLikelihoods = zeros(size(pixelData, 1), numClasses);

for k = 1:numClasses
    mean_k = fuzzyMeans(k, :);
    cov_k = fuzzyCovariances(:,:,k);
    
    % Log of the determinant of the covariance
    logDetCov = log(det(cov_k));
    
    % Inverse of the covariance
    invCov = inv(cov_k);
    
    % Calculate Mahalanobis distance squared (vectorized)
    mahalDistSq = sum(((pixelData - mean_k) * invCov) .* (pixelData - mean_k), 2);
    
    % Discriminant function (log likelihood, ignoring constants)
    logLikelihoods(:, k) = -0.5 * (logDetCov + mahalDistSq);
end

% Assign each pixel to the class with the highest likelihood
[~, classifiedLabels] = max(logLikelihoods, [], 2);
toc;

% --- 5. Visualize and Save the Final Result ---
classifiedImage = reshape(classifiedLabels, rows, cols);

figure('Name', 'Hybrid Fuzzy Classification Results', 'Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
% Use label2rgb to assign a distinct color to each class
rgbLabelImage = label2rgb(classifiedImage, jet(numClasses), 'k', 'shuffle');
imshow(rgbLabelImage);
title('Final Classified Image');
colorbar('Ticks', 1:numClasses, 'TickLabels', "Class " + string(1:numClasses));

% Save the result
imwrite(rgbLabelImage, 'hybrid_classified_image.png');
fprintf('Final classified image saved as "hybrid_classified_image.png"\n');
