function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

fprintf('# of training samples: %f \n', m);
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

% loop over each training sample
for i = 1:m
	% get the row of pixels corresponding to that example
	x = [1; X(i, :)'];
	% compute activation values for each layer until the end
	z_2 = [1; Theta1 * x];
	a_2 = sigmoid(z_2);
	z_3 = Theta2 * a_2;
	a_3 = sigmoid(z_3);
	
	% get index of max value of outputs
	[v, index] = max(a_3);
	p(i) = index;
end





% =========================================================================


end
