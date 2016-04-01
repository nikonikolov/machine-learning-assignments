function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% NB: X is a matrix with each row representing one input entry. Note that
% the number of features does not include X0
% Theta1 and Theta2 contain the parameters for each unit in a row

% Add ones to the X data matrix
X = [ones(m, 1) X];

% z2 contains each training example as a row and the input for each unit a
% column
z2 = transpose(Theta1 * transpose(X));      
a2 = [ones(size(z2,1), 1) sigmoid(z2)];

% z3 contains each training example as a row and the output probability for
% each label as column
z3 = transpose(Theta2 * transpose(a2));      


[vals, p] = max(z3, [], 2); 




% =========================================================================


end
