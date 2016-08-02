function [ L , J] = trainBPBatch( L , input, target, epoch,runs, csi,eta,...
    cost_function,...
    inner_activation, ...
    outter_activation)

%TRAINBPBATCH Summary of this function goes here
%   Detailed explanation goes here

neurons = L.neurons;
normal = 1:length(neurons)+1;
inverted = fliplr(normal);

etaB = eta(1);
etaW = eta(2);
beta = 0.9;
delta = 1e-6;
fprintf('Starting training\n');
% J = zeros(runs,1);
for n=normal
    Delta(n).bias = zeros(size(L(n).bias));
    Delta(n).weight = zeros(size(L(n).weight));
end

begin = 1; i=1; m=1;
while begin
    J(i) = 0;
    for k=normal
        L(k).db = zeros(size(L(k).bias));
        L(k).dW = zeros(size(L(k).weight));
    end;
    
    for ep = 1:epoch
        
            [output,L] = feedforward(L,input(:,m),...
                inner_activation,outter_activation);

            [J_, dJ, err] = cost_function(output,target(:,m));
%             
%             L(1).input = input(m,:); 
%             for n=normal
%               L(n).Y = (L(n).weight)*L(n).input + L(n).bias;
%              [L(n).Z , L(n).dZ] = inner_activation(L(n).Y); L(n+1).input;
%             end
%             [L(n).Z, L(n).dZ] = outter_activation(L(n).Y);
%             L(n+1).input = L(n).Z;
            
%              [J_, dJ, err] = cost_function(target(m,:),L(n).Z);
            
            J(i) = J(i) + J_;
            
%             if J_ > 1000
%                 begin = 0;
%                 break;
%             end
            
            % BP error
            L(length(neurons)+1+1).alpha = dJ;
            L(length(neurons)+1+1).weight = eye(length(err));
            
            for n=inverted
                L(n).alpha = diag(L(n).dZ)*(L(n+1).weight)'*L(n+1).alpha;
                L(n).db = L(n).db + L(n).alpha;
                L(n).dW = L(n).dW + L(n).alpha*(L(n).input)';
            end
            m = m+1; if(m>size(input,2)), m=1; end
    end
    
    for k=normal
        if isempty(L(k).bias)
        else
        Delta(k).bias = etaB*L(k).db + csi*Delta(k).bias;
        L(k).bias = L(k).bias + Delta(k).bias;
        end
        Delta(k).weight = etaW*L(k).dW + csi*Delta(k).weight;
        L(k).weight = L(k).weight + Delta(k).weight;
    end;
    
    J(i) = J(i)/epoch;
    
    if (i>1)
        if(abs(J(i)-J(i-1))/(J(i)) < delta || i > runs)
            begin = 0;
        end
        
        if rem(i,1000) == 0
            i
            etaB = etaB*beta;
            etaW = etaW*beta;
        end
    end
    
    if begin
        i = i+1; if m>size(input,2), m=1; end
%         etaB = etaB*beta;
%         etaW = etaW*beta;
    end
    
end
end

