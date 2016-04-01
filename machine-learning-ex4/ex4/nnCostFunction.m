function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% -------------------------------------------------------------
% Cost Function

% Append bias unit
a1 = vertcat(ones(1,size(X,1)),transpose(X)); 

% In Theta one row represents the parameters for one unit
% 'a' contains weights for training examples as different columns
% 'z' contains the input to g() for each unit in the next layer as columns. 
% Each colums in z represents a different training example
   
z2 = Theta1 * a1;      
a2 = vertcat(ones(1,size(z2,2)),sigmoid(z2)); 

z3 = Theta2 * a2;      
a3 = sigmoid(z3);                       

y_bin = zeros(num_labels, length(y));
for i=1:m
    y_bin(y(i),i) = 1;    
end
    
J = log(a3).*(-y_bin) - log(1-a3).*(1-y_bin);

J=sum(sum(J))/m;

% An easier to understand alternative of the above - rows and columns are reversed
%a1 = [ones(size(X,1), 1) X];

% z2 = transpose(Theta1 * transpose(a1));      
% a2 = [ones(size(z2,1), 1) sigmoid(z2)];   % input for next layer with appended bias unit
% 
% z3 = transpose(Theta2 * transpose(a2));      
% a3 = sigmoid(z3);                         % output

% y_bin = zeros(length(y), num_labels);
% for i=1:m
%     y_bin(i,y(i)) = 1;    
% end
%     
% J = log(a3).*(-y_bin) - log(1-a3).*(1-y_bin);
% 
% J=sum(sum(J))/m;

% -------------------------------------------------------------
% Regularized cost function - remember no regularization for the bias terms
Theta1_sq = Theta1(1:end,2:end).*Theta1(1:end,2:end);
Theta2_sq = Theta2(1:end,2:end).*Theta2(1:end,2:end);

J = J + (sum(sum(Theta1_sq))+sum(sum(Theta2_sq)))*lambda/(2*m);

% -------------------------------------------------------------
% Back propagation

error3 = a3 - y_bin;
error2 = (transpose(Theta2(1:end,2:end))*error3).*sigmoidGradient(z2);

delta3 = error3()*transpose(a2);
delta2 = error2()*transpose(a1);

Theta1_grad = delta2./m;
Theta2_grad = delta3./m;

% Regularized Back propagation

Theta1_grad = Theta1_grad + lambda/m*horzcat(zeros(size(Theta1,1),1), Theta1(1:end,2:end));
Theta2_grad = Theta2_grad + lambda/m*horzcat(zeros(size(Theta2,1),1), Theta2(1:end,2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
