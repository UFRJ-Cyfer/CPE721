function [ J, dJ, err ] = RMSE( output, target )
%RMSE Summary of this function goes here
%   Detailed explanation goes here
err = target-output;
J = err*err'/2;
dJ = err;
end

