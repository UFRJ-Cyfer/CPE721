function [input, target] = extractDataSets (pathname, filename)

[ timeData, controlData ] = openFileMain( pathname, filename );

ind = timeData.ind;
minCurrent = min(timeData.pulseParam(1),timeData.pulseParam(2));
maxCurrent = max(timeData.pulseParam(3),timeData.pulseParam(4));

for k=1:length(timeData.ind)/2
    target(k).Pulse(:,1) = timeData.timeResponse(ind(k*2-1):ind(k*2),1)';
    
    
    
    
    
    current = timeData.timeResponse(ind(2*k-1):ind(k*2),[5]) - ...
        timeData.timeResponse(ind(k*2-1):ind(k*2),[6]);
    current = (current-minCurrent)/maxCurrent;
    
    currentIntegral = 20e-3*cumtrapz(current);
    
    input(k).Pulse(:,1) = current';
    input(k).Pulse(:,2) = currentIntegral';
    input(k).Pulse(:,3) = timeData.timeResponse(ind(k*2-1):ind(k*2),1)';
    input(k).Pulse(:,4) = 20e-3*cumtrapz(input(k).Pulse(:,3))';
    
end

end