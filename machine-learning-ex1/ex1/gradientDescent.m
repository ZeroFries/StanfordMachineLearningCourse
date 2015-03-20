function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


	fprintf('Round %f \n', iter);

	error = zeros(n, 1);
	for j = 1:n
		for i = 1:m
			error(j) += (((theta' * X(i, :)') - y(i)) * X(i, j));
		end
	end
	
	theta = theta - ((alpha/m) * error);
	% theta = tmp_theta;
	
	fprintf('Theta found by gradient descent implementation 1: ');
	fprintf('%f %f \n', theta(1), theta(2));


    % ============================================================

    % Save the cost J in every iteration    
	fprintf('Compute cost: %f \n', computeCost(X, y, theta));
    J_history(iter) = computeCost(X, y, theta);

end

end
