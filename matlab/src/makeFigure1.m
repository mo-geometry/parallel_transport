function makeFigure1(darboux,fig,str)

%% Bloch Sphere 
[kM,jM,iM] = blochSp();

%% The tangent frame
e2=darboux.surfaceFrame(:,4:6); % eTheta
e3=darboux.surfaceFrame(:,7:9); % ePhi

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
m=120;
goldRed = linspace(0.4,1,round(length(r)/m)+2);
otherColour = linspace(0.1,0.85,round(length(r)/m)+2);
ctr=0;

%% Plotting path, basis and spinor
for kk=[1:m:round(length(r)/3),length(r)] %251:25:501
    ctr=ctr+1;
    
%% Extract the tangent vector
    TVecIJK(kk,1) = fig.tangentVector(kk,1)*e2(kk,1) + fig.tangentVector(kk,2)*e3(kk,1); 
    TVecIJK(kk,2) = fig.tangentVector(kk,1)*e2(kk,2) + fig.tangentVector(kk,2)*e3(kk,2);
    TVecIJK(kk,3) = fig.tangentVector(kk,1)*e2(kk,3) + fig.tangentVector(kk,2)*e3(kk,3);

%% The Poles
    plot3(0,0,1,'o','markerfacecolor',[0.98,0.98,0.98],'markersize',8);
    plot3(0,0,-1,'o','markerfacecolor',[0.98,0.98,0.98],'markersize',8);
    
%% The Path - XYZ = KJI
    line(r(1:kk,3),r(1:kk,2),r(1:kk,1),'color','w','LineWidth',3);
    line(r(kk+1:end,3),r(kk+1:end,2),r(kk+1:end,1),'color','w','LineWidth',3);

%% The Tangent Frame - XYZ = KJI
    quiver3(r(kk,3),r(kk,2),r(kk,1),e2(kk,3),e2(kk,2),e2(kk,1),...
        'color','k','LineWidth',3,'MaxHeadSize',0.25,'AutoScaleFactor',0.25)
    quiver3(r(kk,3),r(kk,2),r(kk,1),e3(kk,3),e3(kk,2),e3(kk,1),...
        'color','k','LineWidth',3,'MaxHeadSize',0.25,'AutoScaleFactor',0.25)
    
%% The Tangent Vector - XYZ = KJI
    quiver3(r(kk,3),r(kk,2),r(kk,1),TVecIJK(kk,3),TVecIJK(kk,2),TVecIJK(kk,1),...
        'color',[goldRed(ctr),otherColour(ctr),0.15],'LineWidth',3,'MaxHeadSize',0,'AutoScaleFactor',0.25)
   
%% The Bloch Vector
    quiver3(0,0,0,r(kk,3),r(kk,2),r(kk,1),...
        'color',[1 1 1],'LineWidth',2,'MaxHeadSize',0,'AutoScaleFactor',1)

    drawnow
    
end
