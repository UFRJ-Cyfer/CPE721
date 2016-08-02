function [ output, error, L ] = feedforward( L, input, target, ...
    inner_activation,...
    outter_activation)
%FEEDFORWARD Summary of this function goes here
%   Detailed explanation goes here
neurons = L.neurons;
normal = 1:length(neurons)+1;

L(1).input = input;
for n=normal
    L(n).Y = (L(n).weight)*L(n).input + L(n).bias;
    [L(n).Z, L(n).dZ] = inner_activation(L(n).Y);
    L(n+1).input = L(n).Z;
end
[L(n).Z, L(n).dZ] = outter_activation(L(n).Y);
L(n+1).input = L(n).Z;

output = L(n).Z;
error = target - output;

end

