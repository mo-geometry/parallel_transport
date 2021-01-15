function makeDarboux(dframe,fig,str)

%% Bloch Sphere 
[kM,jM,iM] = blochSp();


%% The tangent frame
e1=dframe(:,1:3);
e2=dframe(:,4:6);
e3=dframe(:,7:9);

%% The bloch vector
r = fig.blochVector;


% %% Preppng the figure
fig1 = figure(1); clf; hold on
set(fig1,'NumberTitle','off','Name',str);

%% The Bloch Sphere
h = surf(kM,jM,iM);
set(h,'FaceAlpha',0.5)
shading interp
colormap winter
axis equal
grid off
axis off
view([160,25]);
hold on

%% Colour vectors
ctr=0;resolution=25;

%% Plotting path, basis and spinor
for kk=[1:resolution:length(r),length(r)] %251:25:501
    ctr=ctr+1;
    
    %% Preppng the figure
    view([fig.thetaPhi(kk,2)*180/pi+90,-5]);
    
%% The poles
    plot3(0,0,1,'o','markerfacecolor',[0.98,0.98,0.98],'markersize',8);
    plot3(0,0,-1,'o','markerfacecolor',[0.98,0.98,0.98],'markersize',8);
    
%% The path - XYZ = KJI
    line(r(1:kk,3),r(1:kk,2),r(1:kk,1),'color','w','LineWidth',2); % [0.4,0.8,0.8]
    line(r(kk+1:end,3),r(kk+1:end,2),r(kk+1:end,1),'color',[0.9 0.9 0.9],'LineWidth',2);

%% The tangent frame - XYZ = KJI
    quiver3(r(kk,3),r(kk,2),r(kk,1),e1(kk,3),e1(kk,2),e1(kk,1),...
        'color','k','LineWidth',2,'MaxHeadSize',0.25,'AutoScaleFactor',0.25)
    quiver3(r(kk,3),r(kk,2),r(kk,1),e2(kk,3),e2(kk,2),e2(kk,1),...
        'color','k','LineWidth',2,'MaxHeadSize',0.25,'AutoScaleFactor',0.25)
    quiver3(r(kk,3),r(kk,2),r(kk,1),e3(kk,3),e3(kk,2),e3(kk,1),...
        'color','k','LineWidth',2,'MaxHeadSize',0.25,'AutoScaleFactor',0.25)

%% The bloch vector - XYZ = KJI
    quiver3(0,0,0,r(kk,3),r(kk,2),r(kk,1),...
        'color',[1 1 1],'LineWidth',1,'MaxHeadSize',0,'AutoScaleFactor',1)
        
    drawnow
    
end

