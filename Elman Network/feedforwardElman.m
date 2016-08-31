function [ output, L, contextInput ] = feedforwardElman( L, dataInput,...
    inner_activation,...
    outter_activation)
%FEEDFORWARD Summary of this function goes here
%   Detailed explanation goes here
neurons = L.neurons;

% if isrow(input)
%     input = input';
% end

% L(n).Y = net values
% L(n).Z = Output
% L(n).dZ = Slope
context = L.context;
L(1).input = dataInput;

    if isempty(L(1).bias)
         L(1).Y = (L(1).weight)*L(1).input;
    else
         L(1).Y = (L(1).weight)*L(1).input + repmat(L(1).bias,1,size(dataInput,2));
    end
    [L(1).Z, L(1).dZ] = inner_activation(L(1).Y);
    
    contextInput = L(1).Z(end-context+1:end);
    L(2).input = L(1).Z;
    
    
    if isempty(L(2).bias)
         L(2).Y = (L(2).weight)*L(2).input;
    else
         L(2).Y = (L(2).weight)*L(2).input + repmat(L(2).bias,1,size(dataInput,2));
    end
    [L(2).Z, L(2).dZ] = outter_activation(L(2).Y);
    L(3).input = L(2).Z;
    output = L(2).Z;
    
end

