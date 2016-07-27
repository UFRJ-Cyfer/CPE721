function [ Z, dZ ] = sigmoid( X )
%SIGMOID Summary of this function goes here
%   Detailed expslanation goes here

    Z = 1./(1+exp(-X));
    dZ = Z.*(eye(size(Z)) - Z);

% Z = tanh(X);
% dZ =  eye(length(X)) - diag(X)^2;
    
end

