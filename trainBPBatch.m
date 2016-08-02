function [ L , J] = trainBPBatch( L , input, target, epoch,csi,eta,...
    cost_function,...
    inner_activation, ...
    outter_activation)

%TRAINBPBATCH Summary of this function goes here
%   Detailed explanation goes here

if isrow(input)
    input = input';
end

if isrow(target)
    target = target';
end

neurons = L.neurons;
normal = 1:length(neurons)+1;
inverted = fliplr(normal);


etaB = eta(1);
etaW = eta(2);
beta = 0.99;
runs = 300;
delta = 1e-5;
fprintf('Starting training\n');

for n=normal
    Delta(n).bias = zeros(size(L(n).db));
    Delta(n).weight = zeros(size(L(n).dW));
end

begin = 1; i=1; m=1;
while begin
    J(i) = 0;
    for k=normal
        L(k).db = zeros(size(L(k).bias));
        L(k).dW = zeros(size(L(k).weight));
    end;
    
    for ep = 1:epoch
        for m=1:length(input)

            [output,~,L] = feedforward(L,input(m,:),target(m,:),...
                inner_activation,outter_activation);

            [J_, dJ, err] = cost_function(target(m,:),output);
            
%             L(1).input = input(m,:); for n=normal
%                 L(n).Y = (L(n).weight)*L(n).input + L(n).bias; [L(n).Z,
%                 L(n).dZ] = inner_activation(L(n).Y); L(n+1).input =
%                 L(n).Z;
%             end [L(n).Z, L(n).dZ] = outter_activation(L(n).Y);
%             L(n+1).input = L(n).Z;
            
% %             [J_, dJ, err] = cost_function(target(m,:),L(n).Z);

            J(i) = J(i) + J_;
            
            if J_ > 1000
                begin = 0;
                break;
            end
            
            % BP error
            L(length(neurons)+1+1).alpha = dJ;
            L(length(neurons)+1+1).weight = eye(length(err));
            
            for n=inverted
                L(n).alpha = diag(L(n).dZ)*(L(n+1).weight)'*L(n+1).alpha;
                L(n).db = L(n).db + L(n).alpha;
                L(n).dW = L(n).dW + L(n).alpha*(L(n).input)';
            end
        end
    end
    
    for k=normal
        if i>1
            L(k).bias = L(k).bias + etaB*L(k).db + csi*Delta(k).bias;
            L(k).weight = L(k).weight + etaW*L(k).dW + csi*Delta(k).weight;
            Delta(k).bias = etaB*L(k).db;
            Delta(k).weight = etaW*L(k).dW;
        else
            Delta(k).bias = etaB*L(k).db;
            Delta(k).weight = etaW*L(k).dW;
            L(k).bias = L(k).bias + etaB*L(k).db;
            L(k).weight = L(k).weight + etaW*L(k).dW;
        end
    end;
    
    J(i) = J(i)/epoch;
    
    if (i>1)
        if(abs(J(i)-J(i-1))/(J(i)) < delta || i > runs)
            begin = 0;
        end
        
        if rem(i,100) == 0
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

