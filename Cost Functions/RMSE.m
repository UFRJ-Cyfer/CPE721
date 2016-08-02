function [ J, dJ, err ] = RMSE( input, target )
%RMSE Summary of this function goes here
%   Detailed explanation goes here
err = target-input;
J = err'*err./2;
dJ = err;
end

