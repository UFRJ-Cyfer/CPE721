function [ L_train , J, J_v] = trainResBPBatch( L ,...
    input, target,...
    valInput,valTarget,...
    runs,...
    csi,eta,...
    cost_function,...
    inner_activation, ...
    outter_activation)

%TRAINBPBATCH Summary of this function goes here
%   Detailed explanation goes here
ValidationChecks = 4;

neurons = L.neurons;
normal = 1:length(neurons)+1;
inverted = fliplr(normal);

etaB = eta(1);
etaW = eta(2);
zeta_plus = 1.2;
zeta_minus = 0.5;
beta = 0.99999;
delta = 1e-6;
deltaMAX = 10;
antigo = struct;
deltaMIN = 0;
a = 1.05;
maxA = 0.2;
a_ = 0.05;
fprintf('Starting training\n');
Jmin = 1e6;
% J = zeros(runs,1);
counter = 0;
for n=normal
    Delta(n).bias = zeros(size(L(n).bias));
    Delta(n).weight = zeros(size(L(n).weight));
    antigo(n).db = zeros(size(L(n).bias));
    antigo(n).dW = zeros(size(L(n).weight));
end

begin = 1; i=1;
while begin

    for k=normal
        L(k).db = zeros(size(L(k).bias));
        L(k).dW = zeros(size(L(k).weight));
    end;
        
        [output,L] = feedforward(L,input,...
            inner_activation, outter_activation);

        [J_, dJ, err] = cost_function(output,target);
        
        [valOutput,~] = feedforward(L,valInput,...
            inner_activation,outter_activation);
        
        [VJ, ~, ~] = cost_function(valOutput,valTarget);
        
%         J_r = regularization(lambda, L);

         J(i) =  J_/size(input,2);
         J_v(i) =  VJ/size(input,2);

    if i>1
       if J_v(i) > 1.05*J_v(i-1)
            counter = counter+1
            Lmin = L;
       else
           counter = 0;
       end
       if counter == ValidationChecks
            begin = 0;
            L_train = Lmin;
            break;
       end
    end
        
        % BP error
        L(length(neurons)+1+1).alpha = dJ;
        L(length(neurons)+1+1).weight = 1;
        
        for n=inverted
            L(n).alpha = (L(n).dZ).*((L(n+1).weight')*L(n+1).alpha);
            L(n).db = L(n).db + sum(L(n).alpha,2);
            
            L(n).dW = L(n).dW + L(n).alpha*L(n).input';
%             L(n).dW = L(n).dW - L(n).weight*lambda;
        end    
    
        for k=normal
            plus = antigo(k).db.*L(k).db > 0;
            minus = antigo(k).db.*L(k).db < 0;
            
            b = antigo(k).db./(antigo(k).db - L(k).db);
            
            a_ = a_*a;
            
%             if(a_ > maxA)
%                 a_ = maxA;
%             end
            
            L(k).db(plus) = L(k).db(plus)*a;
            L(k).db(minus) = L(k).db(minus).*b(minus);
            
            
            plus = antigo(k).dW.*L(k).dW > 0;
            minus = antigo(k).dW.*L(k).dW < 0;
            
            b = antigo(k).dW./(antigo(k).dW - L(k).dW);
            
            L(k).dW(plus) = L(k).dW(plus)*a;
            L(k).dW(minus) = L(k).dW(minus).*b(minus);          
        end
        
        for k =normal
           antigo(k).db = L(k).db;
           antigo(k).dW = L(k).dW;
        end
        
        for n=normal
           L(n).weight = L(n).weight + etaW*L(n).dW/size(input,2);
           L(n).bias = L(n).bias + etaB*L(n).db/size(input,2);
        end
    
   
    
    if (i>1)
%         if(abs(J(i)-J(i-1))/(J(i)) < delta || i > runs)
%             fprintf('NN attended criteria\n')
%             i
%             begin = 0;
%         end

        if(i > runs)
            fprintf('NN attended runs criteria\n')
            L_train = L;
            begin = 0;
        end
        
        if rem(i,1000) == 0
            i
            %             etaB = etaB*beta;
            %             etaW = etaW*beta;
        end
    end
    
    if begin
        i = i+1;
%         etaB = etaB*beta;
%         etaW = etaW*beta;
    end
end
end

