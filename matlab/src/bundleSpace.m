function closedPaths = bundleSpace(q,UofTa,UofTb,UofTc,UofTd,n)

thetaPhi = q.thetaPhi; tvec = q.time;
%% Initialize 
geometricPathA=zeros(n);
dynamicPathA=zeros(n);
globalPathA=zeros(n);
geometricPathB=zeros(n);
dynamicPathB=zeros(n);
globalPathB=zeros(n);
geometricPathC=zeros(n);
dynamicPathC=zeros(n);
globalPathC=zeros(n);
geometricPathD=zeros(n);
dynamicPathD=zeros(n);
globalPathD=zeros(n);

theta=zeros(n);
phi=zeros(n);

fprintf('Generating data .... \n')
%% Extract the S1 fiber bundle of the path
ctr1=0;
for aa=1:round(length(thetaPhi)/n+1):length(thetaPhi)
    ctr1=ctr1+1;%increment
    ctr2=0;%reset
    for bb=1:round(length(thetaPhi)/n+1):length(thetaPhi)
        ctr2=ctr2+1;%increment
        %% Fermion
        temp = bloch( UofTa, tvec, thetaPhi(aa,1), thetaPhi(bb,2) );
        geometricPathA(ctr1,ctr2) = temp.geometricPhase(end);
        dynamicPathA(ctr1,ctr2) = temp.dynamicPhase(end);
        globalPathA(ctr1,ctr2) = temp.globalPhase(end);
        %% Boson 
        temp = bloch( UofTb, tvec, thetaPhi(aa,1), thetaPhi(bb,2) );
        geometricPathB(ctr1,ctr2) = temp.geometricPhase(end);
        dynamicPathB(ctr1,ctr2) = temp.dynamicPhase(end);
        globalPathB(ctr1,ctr2) = temp.globalPhase(end);
        %% Mixed 
        temp = bloch( UofTc, tvec, thetaPhi(aa,1), thetaPhi(bb,2) );
        geometricPathC(ctr1,ctr2) = temp.geometricPhase(end);
        dynamicPathC(ctr1,ctr2) = temp.dynamicPhase(end);
        globalPathC(ctr1,ctr2) = temp.globalPhase(end);
        %% Spin half 
        temp = bloch( UofTd, tvec, thetaPhi(aa,1), thetaPhi(bb,2) );
        geometricPathD(ctr1,ctr2) = temp.geometricPhase(end);
        dynamicPathD(ctr1,ctr2) = temp.dynamicPhase(end);
        globalPathD(ctr1,ctr2) = temp.globalPhase(end);
        
        %% Record the initial state
        theta(ctr1,ctr2) = thetaPhi(aa,1);
        phi(ctr1,ctr2) = thetaPhi(bb,2);
        
        clear temp
    end    
    fprintf('%d/100 complete .... \n',round(aa/length(thetaPhi)*100))
end

%% 
closedPaths.geometricFermion = geometricPathA;
closedPaths.dynamicFermion = dynamicPathA;
closedPaths.globalFermion = globalPathA;
closedPaths.geometricBoson = geometricPathB;
closedPaths.dynamicBoson = dynamicPathB;
closedPaths.globalBoson = globalPathB;
closedPaths.geometricMixed = geometricPathC;
closedPaths.dynamicMixed = dynamicPathC;
closedPaths.globalMixed = globalPathC;
closedPaths.geometricSpinhalf = geometricPathD;
closedPaths.dynamicSpinhalf = dynamicPathD;
closedPaths.globalSpinhalf = globalPathD;
closedPaths.theta = theta;
closedPaths.phi = phi;
