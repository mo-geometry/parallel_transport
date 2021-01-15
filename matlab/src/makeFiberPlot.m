function makeFiberPlot(fig,tvec,str)

%% Generate the moebius band
r = 3;      R = 10;     nr = 20;

globalPhase = fig.globalPhase(1:13:end);
time = linspace(0,2*pi,length(globalPhase));
rho = linspace(-r,r,nr);
 
[TIME,~] = meshgrid(time,rho);
[GlobalPHASE,RHO] = meshgrid(globalPhase,rho);
  
K = (R+RHO.*sin(GlobalPHASE/2)).*cos(TIME);
J = (R+RHO.*sin(GlobalPHASE/2)).*sin(TIME);
I = RHO.*cos(GlobalPHASE/2);

skp=25;

%% Plot the S1 fiber bundle
fig1 = figure(1); clf; hold on
set(fig1,'NumberTitle','off','Name',str);

subplot(1,2,2)
plot(tvec(1:skp:end),fig.geometricPhase(1:skp:end)/pi,'r.','LineWidth',2)
hold on
plot(tvec(1:skp:end),fig.dynamicPhase(1:skp:end)/pi,'g^','LineWidth',2)
plot(tvec(1:skp:end),fig.globalPhase(1:skp:end)/pi,'color',[0.9 0.75 0],'LineWidth',4)
axis tight 
axis square

hold on 
%% Plot the moebius band
subplot(1,2,1)
surf(K,J,I,'EdgeColor','k','facecolor','interp')
colormap copper
alpha(0.35)
grid off
shading interp
axis('equal')
axis off