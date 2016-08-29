function [ ] = plotNNResults( L,J, input, target, ...
    inner_activation, outter_activation)
%PLOTNNRESULTS Summary of this function goes here
%   Detailed explanation goes here
close all;
neurons = L.neurons;

err = zeros(size(target));
% output = zeros(size()); % caso simples de aproximador universal

    for m=1:size(input,2)
        [output(m),L] = feedforward(L,input(:,m),...
            inner_activation,outter_activation);
        if m < length(target)
            err(m) = target(:,m) - output(m);
        end
    end
    
    
    subplot(3,1,1)
    plot(output,'r'); 
    hold on;
     plot(target,'k--');  
%   legend('NN Input','NN Output','Target');
    legend('NN Output','Target');
    
    subplot(3,1,2);
    plot(err,'r')
    legend('Error')
    hold on;
%   plot(input*sstd+smean - (output*std(ax)+mean(ax)+0.05)','k');
%   legend('Error clean sinc','error noised sinc');
    
    subplot(3,1,3)
    plot(J);
    legend('Cost');
    
    figure;
    plot(output)
end

