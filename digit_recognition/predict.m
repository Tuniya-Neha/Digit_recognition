function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

X=[ones(m,1), X];
Z=Theta1*X';
a=sigmoid(Z);
t=a';
t=[ones(m,1),t]
a=t';
%a=[ones(1,m),a];
h=Theta2*a;
k=sigmoid(h);
[val, idx]=max(k,[],1);
p=idx';








% =========================================================================


end
