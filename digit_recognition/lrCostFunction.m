function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
z=X*theta;
h=sigmoid(z);
sqrtheta=theta.^2;
%sumsqrtheta=sum(sqrtheta)-theta(1).^2;

J=(1/m)*((-y'*log(h))-(1-y')*log(1-h))+((lambda/(2*m))*sum(theta(2:end).^2));
prediction=h-y;

grad=(1/m)*(X'*prediction)+((lambda/m)*theta);

grad(1)=grad(1)-(lambda/m)*theta(1);

grad = grad(:);

end
