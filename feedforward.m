function [ output, L ] = feedforward( L, input,...
    inner_activation,...
    outter_activation)
%FEEDFORWARD Summary of this function goes here
%   Detailed explanation goes here
neurons = L.neurons;
normal = 1:length(neurons)+1;

if isrow(input)
    input = input';
end

% L(n).Y = net values
% L(n).Z = Output
% L(n).dZ = Slope
L(1).input = input;
for n=normal
    if isempty(L(n).bias)
         L(n).Y = (L(n).weight)*L(n).input;
    else
         L(n).Y = (L(n).weight)*L(n).input + L(n).bias;
    end
    [L(n).Z, L(n).dZ] = inner_activation(L(n).Y);
    L(n+1).input = L(n).Z;
end
[L(n).Z, L(n).dZ] = outter_activation(L(n).Y);
L(n+1).input = L(n).Z;

output = L(n).Z;

end

