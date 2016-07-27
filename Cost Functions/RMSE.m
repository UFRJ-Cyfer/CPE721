function [ J, dJ, err ] = RMSE( input, output )
%RMSE Summary of this function goes here
%   Detailed explanation goes here
err = output-input;
J = err'*err/2;
dJ = err;
end

