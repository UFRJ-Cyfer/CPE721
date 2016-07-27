function [L]= NNGenerator( neurons, input, target,...
                              epoch, cost_function,...
                              inner_activation, outter_activation)
%NNGENERATOR Summary of this function goes here
%   Detailed explanation goes here

if isrow(input)
    input = input';
end
if isrow(target)
    target = target';
end
span = 0.2;
etaB = 1e-4;
etaW = 1/2;
beta = 1;
runs = 500;
delta = 1e-8;

column = length(input);
size_output = length(target);

column = 1;
size_output = 1;
% neurons is a variable that holds the number of neurons of each
% hidden layer, therefore, specifying neurons = [5 4 7] creates a NN
% with 3 hidden layers, each with 5, 4 , and 7 neurons respectively.
L(1).weight = -span + 2*span*rand(neurons(1),column);
L(1).bias = -span + 2*span*rand(neurons(1),1);
for n = 2:length(neurons)
    L(n).weight = -span + 2*span*rand(neurons(n),neurons(n-1));
    L(n).bias = -span + 2*span*rand(neurons(n),1);
end
L(length(neurons)+1).weight = -span + 2*span*rand(size_output,neurons(end));
L(length(neurons)+1).bias = -span + 2*span*rand(size_output,1);
begin = 1; i=1; m=1;

L(1).input = input(m,:);
for n=1:length(neurons)+1
     L(n).Y = zeros(size((L(n).weight)*L(n).input + L(n).bias));
    (L(n).weight)*L(n).input + L(n).bias;
    (L(n).weight)*L(n).input + L(n).bias;
    [L(n).Z, L(n).dZ] = tgh(L(n).Y);
    L(n+1).input = L(n).Z;
end

for n=fliplr(1:length(neurons)+1)
    L(n).alpha = zeros(size(target(m,:)));
    L(n).db = zeros(size(L(n).bias));
    L(n).dW = zeros(size(L(n).weight))';
end
normal = 1:length(neurons)+1;            
inverted = fliplr(normal);

inputDummy = reshape(input,10, []);
targetDummy = reshape(target,10,[]);
targetDummyplot = target;
inputDummyplot = input;

for slot=1:length(input)/10
    input = inputDummy(slot);
    target = targetDummy(slot);
J(1) =0;
    while begin
        J(i) = 0;
        for k=normal
            L(k).db = zeros(size(L(k).bias));
            L(k).dW = zeros(size(L(k).weight));
        end;
        
        for ep = 1:epoch
                for m=1:length(input)
                    L(1).input = input(m,:);
                for n=normal
                     L(n).Y = (L(n).weight)*L(n).input + L(n).bias;
                    [L(n).Z, L(n).dZ] = inner_activation(L(n).Y);
                    L(n+1).input = L(n).Z;
                end
                    [L(n).Z, L(n).dZ] = outter_activation(L(n).Y);
                    L(n+1).input = L(n).Z;
                    
                [J_, dJ, err] = cost_function(target(m,:),L(n).Z);
                J(i) = J(i) + J_;
                if J_ > 1000
                    begin = 0;
                    break;
                end

                % BP error
                    L(n+1).alpha = dJ;
                    L(n+1).weight = eye(length(err));

                for n=inverted
                    L(n).alpha = diag(L(n).dZ)*(L(n+1).weight)'*L(n+1).alpha;
                    L(n).db = L(n).db + L(n).alpha;
                    L(n).dW = L(n).dW + L(n).alpha*(L(n).input)';
                end
            end
        end
        
        for k=normal
            L(k).bias = L(k).bias + etaB*L(k).db;
            L(k).weight = L(k).weight + etaW*L(k).dW;
        end;

        J(i) = J(i)/epoch;
        if (i>1)
%             if(abs(J(i)-J(i-1))/(J(i)) < delta || i > runs)
%                 if J(i) < delta^2
%                     begin = 0;
%                 end
%             end
            if i>runs
                begin =0;
            end
            if rem(i,1000) == 0
                i
            end
        end

        if begin
            i = i+1; if m>length(input), m=1; end
            etaB = etaB*beta;
            etaW = etaW*beta;
        end            

    end
end
    
    
    hold on;
    for m=1:length(inputDummyplot)
        L(1).input = inputDummyplot(m,:);
        for n=1:length(neurons)+1
            L(n).Y = (L(n).weight)*L(n).input + L(n).bias;
            [L(n).Z, L(n).dZ] = inner_activation(L(n).Y);
            L(n+1).input = L(n).Z;
        end
       [output_plot(m), ~] = outter_activation(L(n).Y);
    end
    plot(output_plot,'r'); 
    plot(targetDummyplot,'ko');  
    figure; plot(J);

end









