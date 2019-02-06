function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

h = sigmoid(X*theta);
pt1 = -y'*(log(h));
pt2 = (1-y')*(log(1-h));
sqrerr = sum(pt1 - pt2);
theta(1) = 0;
theta_sqr = theta'*theta;
reg_expr = (lambda/(2*m)) * theta_sqr;
J = ((1/m)*sqrerr) + reg_expr;

errors_vec = X' * (h-y);
theta(1) = 0;
reg_term = (lambda/m) * theta;
grad = (1/m * errors_vec) + reg_term;



% =============================================================

end
