function [ L_train , J] = trainBPBatch( L , input, target, epoch,runs,...
    csi,eta,lambda,...
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
zeta_plus = 1.2;
zeta_minus = 0.5;
beta = 0.9999;
delta = 1e-6;
deltaMAX = 10;
deltaMIN = 0;

fprintf('Starting training\n');
Jmin = 1e6;
% J = zeros(runs,1);

for n=normal
    Delta(n).bias = zeros(size(L(n).bias));
    Delta(n).weight = zeros(size(L(n).weight));
end

begin = 1; i=1;
antigo = 0;
while begin
    J(i) = 0;
    for k=normal
        L(k).db = zeros(size(L(k).bias));
        L(k).dW = zeros(size(L(k).weight));

    end;
    dJ = 0;
    J(i) = 0;
        
        
        [output,L] = feedforward(L,input,...
            inner_activation, outter_activation);
        
        %             input(:,m) = output;
        
        [J_, dJ, err] = cost_function(output,target);
        
        
        J_r = regularization(lambda, L);

         J(i) = J(i) + J_ + J_r;
%         
%         if J(i) < Jmin
%             Jmin = J(i);
%             L_train = L;
%         end
        
%         if J(i)/epoch > 100000
%             fprintf('NN Exploded\n')
%             begin = 0;
%             break;
%         end
        
        % BP error
        L(length(neurons)+1+1).alpha = dJ;
        L(length(neurons)+1+1).weight = 1;
        
        for n=inverted
            L(n).alpha = (L(n).dZ).*((L(n+1).weight')*L(n+1).alpha);
            L(n).db = L(n).db + sum(L(n).alpha,2);
            
            L(n).dW = L(n).dW + L(n).alpha*L(n).input';
%             L(n).dW = L(n).dW - L(n).weight*lambda;
        end
        
        
        for n=normal
           L(n).weight = L(n).weight + etaW*L(n).dW/size(input,2);
           L(n).bias = L(n).bias + etaB*L(n).db/size(input,2);
        end

%     for k=normal
%         if isempty(L(k).bias)
%         else
%             aux = (Delta(k).bias.*L(k).db) > 0;
%             Delta(k).bias(aux) = min(Delta(k).bias(aux)*zeta_plus,deltaMAX);
%             Delta(k).bias(aux) = -sign(L(k).db(aux)).*L(k).db(aux);
%             
%             aux = (Delta(k).bias.*L(k).db) < 0;
%             Delta(k).bias(aux) = max(Delta(k).bias(aux)*zeta_minus,deltaMIN);
%  if i>1
%             if J(i) > J(i-1)
%                 L(k).bias = L(k).bias - Delta(k).bias;
%             end
%  end     
%             L(k).db(aux) = 0;
%             
%             aux = (Delta(k).bias.*L(k).db) == 0;
%             Delta(k).bias(aux) = -sign(L(k).db(aux)).*L(k).db(aux);
%             
%             L(k).bias(aux) = L(k).bias(aux) - Delta(k).bias(aux)*etaB;
%             
%         end
%             aux = (Delta(k).weight.*L(k).dW) > 0;
%             Delta(k).weight(aux) = min(Delta(k).weight(aux)*zeta_plus,deltaMAX);
%             Delta(k).weight(aux) = -sign(L(k).dW(aux)).*L(k).weight(aux);
%             
%             aux = (Delta(k).weight.*L(k).dW) < 0;
%             Delta(k).weight(aux) = max(Delta(k).weight(aux)*zeta_minus,deltaMIN);
%  if i>1
%             if J(i) > J(i-1)
%                 L(k).weight(aux) = L(k).weight(aux) - Delta(k).weight(aux);
%             end
%  end         
%             L(k).dW(aux) = 0;
%             
%             aux = (Delta(k).weight.*L(k).dW) == 0;
%             Delta(k).weight(aux) = -sign(L(k).dW(aux)).*L(k).dW(aux);
%             
%             L(k).weight(aux) = L(k).weight(aux) - Delta(k).weight(aux)*etaW;
%         
%     end;
%     
%     for k = normal
%         L(k).weight(abs(L(k).weight)>10) = 0;
%         L(k).bias(abs(L(k).bias)>10) = 0;
%     end
%     
    
    J(i) = J(i)/epoch;
    
    if (i>1)
%         if(abs(J(i)-J(i-1))/(J(i)) < delta || i > runs)
%             fprintf('NN attended criteria\n')
%             begin = 0;
%         end

        if(i > runs)
            fprintf('NN attended criteria\n')
            begin = 0;
        end
        
        if rem(i,100) == 0
            i
            %             etaB = etaB*beta;
            %             etaW = etaW*beta;
        end
    end
    
    if begin
        i = i+1;
        etaB = etaB*beta;
        etaW = etaW*beta;
    end
end
L_train = L;
end

