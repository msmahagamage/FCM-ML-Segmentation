function [clusterCenters, membershipMatrix, objectiveHistory] = fcmcluster(data, numClusters, options)

% --- 1. Argument Validation and Options Setup ---
if nargin < 2
    error('FCM requires at least two arguments: data and numClusters.');
end
if nargin < 3
    options = struct(); % Use default options if none are provided
end

% Set default options
defaultOptions = struct('fuzziness', 2.0, 'maxIter', 100, 'tolerance', 1e-5, 'verbose', true);
optionNames = fieldnames(defaultOptions);
for i = 1:length(optionNames)
    if ~isfield(options, optionNames{i})
        options.(optionNames{i}) = defaultOptions.(optionNames{i});
    end
end

% Validate inputs
validateattributes(data, {'numeric'}, {'nonempty', '2d'}, mfilename, 'data');
validateattributes(numClusters, {'numeric'}, {'scalar', 'integer', '>', 1}, mfilename, 'numClusters');
if size(data, 1) < numClusters
    error('Number of data points must be greater than the number of clusters.');
end
validateattributes(options.fuzziness, {'numeric'}, {'scalar', '>', 1}, mfilename, 'options.fuzziness');
validateattributes(options.maxIter, {'numeric'}, {'scalar', 'integer', '>', 0}, mfilename, 'options.maxIter');
validateattributes(options.tolerance, {'numeric'}, {'scalar', 'positive'}, mfilename, 'options.tolerance');


% --- 2. Initialization ---
[numPoints, ~] = size(data);
objectiveHistory = zeros(options.maxIter, 1);

% Initialize membership matrix U with random values
membershipMatrix = rand(numClusters, numPoints);
colSum = sum(membershipMatrix, 1);
membershipMatrix = membershipMatrix ./ colSum; % Ensure columns sum to 1

% --- 3. Main Iteration Loop (Vectorized) ---
for iter = 1:options.maxIter
    mf = membershipMatrix.^options.fuzziness;
    
    % Vectorized center calculation
    clusterCenters = (mf * data) ./ sum(mf, 2);
    
    % Vectorized distance calculation
    dist = pdist2(clusterCenters, data) + 1e-9; % Add epsilon to avoid division by zero
    
    % Vectorized membership calculation
    tmp = (dist.^(-2)).^(1 / (options.fuzziness - 1));
    newMembership = tmp ./ sum(tmp, 1);
    
    % Calculate Objective Function to check for convergence
    objectiveHistory(iter) = sum(sum((dist.^2) .* mf));
    
    % Check for convergence
    if iter > 1
        if abs(objectiveHistory(iter) - objectiveHistory(iter-1)) < options.tolerance
            if options.verbose
                fprintf('Converged after %d iterations.\n', iter);
            end
            break;
        end
    end
    
    if options.verbose
        fprintf('Iteration %d: Objective Function = %f\n', iter, objectiveHistory(iter));
    end
    
    membershipMatrix = newMembership;
end

% Trim unused history if convergence was early
objectiveHistory(iter+1:end) = [];

if iter == options.maxIter && options.verbose
    fprintf('Reached maximum number of iterations (%d).\n', options.maxIter);
end

end
