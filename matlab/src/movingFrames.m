function darboux = movingFrames(r,thetaPhi,tvec)

%% TIME INTERVAL; SCALAR COORDINATES 
Dt = tvec(2)-tvec(1);
theta = thetaPhi(:,1);
phi = thetaPhi(:,2);

%% Velocity of the bloch vector
di = gradient(r(:,1),Dt);
dj = gradient(r(:,2),Dt);
dk = gradient(r(:,3),Dt);
dr = [di dj dk];

%% Unit tangent vector
eT = dr./mag(dr,3);

%% Gradient of the tangent vector 
dTi =  gradient(eT(:,1),Dt);
dTj =  gradient(eT(:,2),Dt);
dTk =  gradient(eT(:,3),Dt);
dT = [dTi dTj dTk];

%% Unit center of force vector
eF = dT./mag(dT,3);
%% Unit binormal vector
eB = cross(eT,eF);
%% Unit surface normal (2-sphere)
eN = [cos(theta) sin(theta).*sin(phi) sin(theta).*cos(phi)];
%% Unit tangent-normal vector (2-sphere)
eTN = cross(eT,eN);
%% Unit azimuthal vector (2-sphere)
eTh = [-sin(theta) cos(theta).*sin(phi) cos(theta).*cos(phi)];
%% Unit polar vector (2-sphere)
eP = [ zeros(size(phi)) cos(phi) -sin(phi)];

%% The moving frames
darboux.surfaceFrame = [eN eTh eP];
darboux.curveFrame = [eN eT eTN];
darboux.frenetSerretFrame =[eF eT eB];

function N = mag(T,n)
% MAGNATUDE OF A VECTOR (Nx3)
%  M = mag(U)
N = sum(abs(T).^2,2).^(1/2);
d = find(N==0); 
N(d) = eps*ones(size(d));
N = N(:,ones(n,1));