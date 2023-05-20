close all;
clear all;

n = 10; % number of variables
s = 4; % number of subsystems

inits = 1000; % number of random initialization points
max_iter = 1000; % maximum number of iterates to be run

noise_bound = 1;

randn('state',1); % set state so problem is reproducable
rand('state',1); % set state so problem is reproducable

% generate a problem instances
A = randn(n,n,s);
for i = 1:s
    A(:,:,i) = transpose(A(:,:,i))*A(:,:,i);
end
b = randn(n,s);
%%

L_i = zeros(s,1);
ell_i = zeros(s,1);

% calculate Lipchitz and strong convexity constants using maximum and
% minimum eigen values of A^T.A 
for i = 1:s
    offset = (i-1)*n;
    L_i(i) = 2*max(eig(A(:,:,i)));
    ell_i(i) = 2*min(eig(A(:,:,i)));
end
L = sum(L_i); % Lipchitz constant of the whole function
ell = sum(ell_i); % Strong convexity constant of the whole function
p = 0.5;
gamma = p/L;
gamma_ell = gamma*ell;

r_max = (sqrt(ell))/(sqrt(L)+sqrt(ell));

%%
%solve problem using matlab optimization library to approximate x^{*}
options = optimoptions('fminunc','Display', 'off', 'MaxFunctionEvaluations', 10000);
x_min_0 = zeros(n,1);
obj_fun = @(x) 0;
for i =1:s
    obj_fun = @(x) obj_fun(x) + transpose(x)*A(:,:,i)*x + transpose(b(:,i))*x;
end

grad_fun = @(x) 0;
for i =1:s
    grad_fun = @(x) grad_fun(x) + 2*A(:,:,i)*x + b(:,i);
end

opt_x_min = fminunc(obj_fun,x_min_0,options); % optimal value of x^{*}
opt_val= obj_fun(opt_x_min); %  minimum value of the function


%%
% generating random initialization points
x_inits = zeros(n,inits);
for i = 1:inits
    x_inits(:,i) = 1000*randn(n,1);
end

% set r and epsilon
r = 0.03;
epsilon_vals =[0.01,0.1,1,10];

max_resets = 400; % to create variables to collect data for plotting purposes 
func_values_reset = zeros(max_resets,2*numel(epsilon_vals),inits);

% create variables to collect data durring optimization
x_gaps = zeros(max_iter+1,s,2*numel(epsilon_vals),inits);
function_values = zeros(max_iter+1,s,2*numel(epsilon_vals),inits);
grad_values = zeros(max_iter+1,s,2*numel(epsilon_vals),inits);
resets = zeros(max_iter,numel(epsilon_vals)+1,inits);
resets(:,end,:) = ones(max_iter,1,inits);

x_gaps_ideal = zeros(max_iter+1,s,inits);
function_values_ideal = zeros(max_iter+1,s,inits);
grad_values_ideal = zeros(max_iter+1,s,inits);

% solve the problem for different initialization points
for n_x = 1:inits
    
    % set initialization point
    x_init = x_inits(:,n_x);
    
    % solve the problem in distributed setting
    for n_k = 1:numel(epsilon_vals)
        [x_gaps(:,:,n_k,n_x),function_values(:,:,n_k,n_x),grad_values(:,:,n_k,n_x),resets(:,n_k,n_x)]  = solver(max_iter,n,s,A,b,epsilon_vals(n_k),obj_fun,grad_fun,gamma,x_init,opt_x_min, r); % with imperfect gradient communication and IntSync Step
        [x_gaps(:,:,numel(epsilon_vals)+n_k,n_x),function_values(:,:,numel(epsilon_vals)+n_k,n_x),grad_values(:,:,numel(epsilon_vals)+n_k,n_x)]  = solver(max_iter,n,s,A,b,epsilon_vals(n_k),obj_fun,grad_fun,gamma,x_init,opt_x_min, 0); % with IGDDS (set r=0)
    end
    
    [x_gaps_ideal(:,:,n_x),function_values_ideal(:,:,n_x),grad_values_ideal(:,:,n_x)]  = solver(max_iter,n,s,A,b,0,obj_fun,grad_fun,gamma,x_init,opt_x_min, 0); % with perfect communication (set r = 0 and epsilon =0)
    
    % extract readings in Int Sync steps for plotting purposes
    for i= 1:numel(epsilon_vals)
        func_values_reset_raw = function_values(resets(:,i,n_x) == 1,1,i,n_x);
        func_values_reset(:,i,n_x) = func_values_reset_raw(1:max_resets);
        func_values_reset(:,numel(epsilon_vals)+i,n_x) = function_values(1:max_resets,1,numel(epsilon_vals)+i,n_x);
        
    end  
end

%%
itrs = 0:max_iter;

% average the readings for all initialization points
x_gaps = mean(x_gaps, 4);
function_values = mean(function_values, 4);
grad_values = mean(grad_values, 4);
resets = mean(resets,3);
func_values_reset = mean(func_values_reset,3);

x_gaps_ideal = mean(x_gaps_ideal, 3);
function_values_ideal = mean(function_values_ideal, 3);
grad_values_ideal = mean(grad_values_ideal, 3);

% get reset counts
reset_counts = zeros(max_iter+1, numel(epsilon_vals)+1);
for i = 1:max_iter
    reset_counts(i+1,:) = sum(resets(1:i,:));
end

save('results.mat')

