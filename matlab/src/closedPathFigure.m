function closedPathFigure(closedPaths,str1,str2,str3)

fprintf('Displaying figure #2 - The global phase of the closed path ... \n');
%% Plot figure #2
fig1 = figure(1); clf; hold on
set(fig1,'NumberTitle','off','Name',str1);
subplot(1,2,1)
surf(closedPaths.theta/pi,closedPaths.phi/pi,closedPaths.globalFermion/pi);
view([55 10])
subplot(1,2,2)
surf(closedPaths.theta/pi,closedPaths.phi/pi,closedPaths.globalBoson/pi);
view([235 15])
colormap summer
axis square
askContinue('Press enter to generate figure #3')

fprintf('Displaying figure #3 - The global and dynamics phase of the closed path ... \n');
%% Plot figure #3
fig1 = figure(1); clf; hold on
set(fig1,'NumberTitle','off','Name',str2);
subplot(1,2,1)
surf(closedPaths.theta/pi,closedPaths.phi/pi,closedPaths.globalMixed/pi);
view([235 15])
subplot(1,2,2)
surf(closedPaths.theta/pi,closedPaths.phi/pi,closedPaths.dynamicMixed/pi);
view([235 15])
colormap summer
axis square
askContinue('Press enter to generate figure #6')

fprintf('Displaying figure #6 - The global phase of the spin half unitary ... \n');
%% Plot figure #6
fig1 = figure(1); clf; hold on
set(fig1,'NumberTitle','off','Name',str3);
surf(closedPaths.theta/pi,closedPaths.phi/pi,closedPaths.globalSpinhalf/pi);
view([235 15])
colormap summer
axis square
askContinue('Press enter to finish.')