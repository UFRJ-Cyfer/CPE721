function [ J_r ] = regularization( lambda, L )
%REGULARIZATION Summary of this function goes here
%   Detailed explanation goes here
    J_r = 0;
    for k=1:length([L.neurons])
         E = (L(k).weight)'*L(k).weight;
         J_r = J_r + trace(E);
    end
    
    J_r = J_r*lambda/2;
    
end

