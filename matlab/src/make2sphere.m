function make2sphere(darboux,fig,str)

%% Bloch Sphere 
[kM,jM,iM] = blochSp();

%% The tangent frame
e2=darboux.surfaceFrame(:,4:6);
e3=darboux.surfaceFrame(:,7:9);

%% The bloch vector
r = fig.blochVector;

%% Initialize the tangent vector
TVecIJK=zeros(length(r),3);

%% Preppng the figure
fig1 = figure(1); clf; hold on
set(fig1,'NumberTitle','off','Name',str);

%% The Bloch Sphere
h = surf(kM,jM,iM);
set(h,'FaceAlpha',0.35)
shading interp
colormap winter
axis equal
grid off
axis off
view([160,25]);

%% Colour vectors
ctr=0; resolution=25;
goldRed = linspace(0.4,1,round(length(1:resolution:length(r)))+1);
otherColour = linspace(0.1,0.85,round(length(1:resolution:length(r)))+1);

%% Plotting path, basis and spinor
for kk=[1:resolution:length(r),length(r)] 
    ctr=ctr+1;
    %% Preppng the figure
    view([fig.thetaPhi(kk,2)*180/pi+90,-5]);
    
%% Extract the tangent vector
    TVecIJK(kk,1) = fig.tangentVector(kk,1)*e2(kk,1)+fig.tangentVector(kk,2)*e3(kk,1); 
    TVecIJK(kk,2) = fig.tangentVector(kk,1)*e2(kk,2)+fig.tangentVector(kk,2)*e3(kk,2);
    TVecIJK(kk,3) = fig.tangentVector(kk,1)*e2(kk,3)+fig.tangentVector(kk,2)*e3(kk,3);

%% The poles
    plot3(0,0,1,'o','markerfacecolor',[0.98,0.98,0.98],'markersize',8);
    plot3(0,0,-1,'o','markerfacecolor',[0.98,0.98,0.98],'markersize',8);
    
%% The path - XYZ = KJI
    line(r(1:kk,3),r(1:kk,2),r(1:kk,1),'color','w','LineWidth',2); 
    line(r(kk+1:end,3),r(kk+1:end,2),r(kk+1:end,1),'color',[0.8 0.8 0.8],'LineWidth',2);

%% The bloch vector - XYZ = KJI
    quiver3(0,0,0,r(kk,3),r(kk,2),r(kk,1),...
        'color',[1 1 1],'LineWidth',1,'MaxHeadSize',0,'AutoScaleFactor',1)
    
%% The tangent vector - XYZ = KJI
    quiver3(r(kk,3),r(kk,2),r(kk,1),TVecIJK(kk,3),TVecIJK(kk,2),TVecIJK(kk,1),...
        'color',[goldRed(ctr),otherColour(ctr),0.15],'LineWidth',2,'MaxHeadSize',0,'AutoScaleFactor',0.25)
    drawnow
end
