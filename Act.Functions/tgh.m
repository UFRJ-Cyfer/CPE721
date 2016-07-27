function [ Z, dZ ] = tgh( X )
%TGH Summary of this function goes here
%   Detailed explanation goes here

Z = 2./(1+exp(-2*X))-1;
dZ = 1 - Z.^2;
end

