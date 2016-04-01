function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta




ThetaX = transpose(theta)*transpose(X);     % row vector
Htheta = transpose(sigmoid(ThetaX));        % column vector
theta_SQ = theta.^2;
ltheta = length(theta);

% J results as a vector
J = -y.*log(Htheta) - (1-y).*log(1-Htheta);

% J is now a scalar
J = 1/m*(sum(J) + lambda/2*(sum(theta_SQ(2:ltheta))));

grad = 1/m*(transpose(X)*(Htheta-y) + lambda*vertcat(0, theta(2:ltheta)));     % column vector

% =============================================================

end
