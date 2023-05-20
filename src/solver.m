function [x_gaps, function_values,grad_values, resets]  = solver(max_iter,n,s,A,b,epsilon,obj_fun,grad_fun,gamma,x_init,opt_x_min,r)

x_0 = x_init; % set initialization point

% create a 2-D vector to store local copies of variable
x = zeros(n,s);
for i = 1:s
    x(:,i) = x_0;
end

% variables to store readings durring optimization
x_gaps = zeros(max_iter+1,s);
grad_values = zeros(max_iter+1,s);
function_values = zeros(max_iter+1,s);
gradients = zeros(n,s);
resets = zeros(max_iter,1);

% initialize variables
reset = 0;
rst = 0;
grads_comp = zeros(n,s);
i = 0;

%solve the problem distributed setting
while i < max_iter
    
    % calcualate function values, gradient norms, gradients and distance to optimal
    % solution
    for j=1:s
        function_values(i+1,j) = obj_fun(x(:,j)); % function values
        grad_values(i+1,j) = norm(grad_fun(x(:,j))); % gradient norms
        gradients(:,j) = 2*A(:,:,j)*x(:,j) + b(:,j); % gradients
        x_gaps(i+1,j) = norm(x(:,j) - opt_x_min); % distance to optimal solution
    end
    for j = 1:s
        %add noise to the gradient communications of the subsystems
        
        grads_j = zeros(n,s,s);
        for k=1:s
            if j~=k
                noise = randn(n,1); % generate noise randomly
                noise = noise/norm(noise); % normalize randomly generated noise
                fact = (randi(100))/100; % generate random number between 0-1
                noise_jk = epsilon*fact*noise; % multiply normalized random noise vector by factor*epsilon to get bounded noise vector
                grads_j(:,j,k) = gradients(:,k) + noise_jk; % add noise to gradients
            else
                grads_j(:,j,k) = gradients(:,k); % no noise added for the self gradient
            end
        end
        
        % Calculate gradient estimations at subsystem
        grads_comp(:,j) = zeros(n,1);
        for k=1:s
            grads_comp(:,j) = grads_comp(:,j) + grads_j(:,j,k);
        end
        % Trigger IntSync if the relative error measure is high
        if r*norm(grads_comp(:,j)) <= 2*(epsilon*s)*(i-rst+0.5)
            reset = 1;
        end
    end
    
    % Perform GD update and IntSync
    if reset == 1
        if rst == i % State 2
            for j=1:s
                x(:,j) = x(:,j) - gamma*grads_comp(:,j); % GD update
            end
            i = i+1;
        end
        % IntSync step
        x_c = sum(x,2)/s; 
        for j=1:s
            x(:,j) = x_c;
        end
        resets(i) = 1;
        rst = i;
        reset = 0;
    else
        for j=1:s
            x(:,j) = x(:,j) - gamma*grads_comp(:,j); % GD Update
        end
        i = i + 1;
    end
end

end