function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

%   X is a matrix with rows representing training examples and columns
%   representing training features.
%   theta is a column vector representing the current values of the parameters
%   y is a column vector representing the results we have

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

ThetaX = transpose(theta)*transpose(X);     % row vector
Htheta = transpose(sigmoid(ThetaX));        % column vector

% J results as a vector
J = -y.*log(Htheta) - (1-y).*log(1-Htheta);

% J is now a scalar
J = 1/m*sum(J);

grad = 1/m* transpose(X)*(Htheta - y);     % column vector



% =============================================================

end
