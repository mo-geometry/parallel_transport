function fig = bloch(UofT, tvec, theta0, phi0)

%% Extracting the quaternion
Dt = tvec(2)-tvec(1); % time step
a=UofT(:,1);b=UofT(:,2);c=UofT(:,3);d=UofT(:,4); % quaternion elements

%% Derivative of the quaternion
da = gradient(a,Dt);
db = gradient(b,Dt);
dc = gradient(c,Dt);
dd = gradient(d,Dt);

%% Evolving the bloch vector rN
r0 = [cos(theta0);sin(theta0)*sin(phi0);sin(theta0)*cos(phi0)];
fig.blochVector = zeros(length(tvec),3);

for kk=1:length(tvec)    
    Rotor=[ a(kk)^2+b(kk)^2-c(kk)^2-d(kk)^2,    2*(b(kk)*c(kk)-a(kk)*d(kk)),        2*(b(kk)*d(kk)+a(kk)*c(kk));
            2*(b(kk)*c(kk)+a(kk)*d(kk)),        a(kk)^2-b(kk)^2+c(kk)^2-d(kk)^2,    2*(c(kk)*d(kk)-a(kk)*b(kk));
            2*(b(kk)*d(kk)-a(kk)*c(kk)),        2*(c(kk)*d(kk)+a(kk)*b(kk)),        a(kk)^2-b(kk)^2-c(kk)^2+d(kk)^2];
	fig.blochVector(kk,:)=Rotor*r0;
end

%% Hamiltonian Elements
Hi = 2*(a.*db - da.*b - dc.*d + c.*dd);
Hj = 2*(a.*dc - da.*c + db.*d - b.*dd);
Hk = 2*(a.*dd - da.*d - db.*c + b.*dc);

%% The bloch vector
ri = fig.blochVector(:,1);
rj = fig.blochVector(:,2);
rk = fig.blochVector(:,3);

%% Phase Relations
Dtheta = (Hk.*rj-Hj.*rk)./sqrt(rk.^2+rj.^2);
Dphi = -Hi + (Hk.*rk+Hj.*rj)./(rk.^2+rj.^2).*ri;
Dgeoph = Dphi.*ri;
Dgph = (Hk.*rk+Hj.*rj)./(rk.^2+rj.^2);
Ddyn = Hi.*ri+Hj.*rj+Hk.*rk;

%% Initialisation
phi = zeros(size(Dphi));
theta = zeros(size(Dtheta));
fig.geometricPhase = zeros(size(Dgeoph));
fig.globalPhase = zeros(size(Dgph));
fig.dynamicPhase = zeros(size(Ddyn)); 

%% Integrating the angles
for kk=1:length(tvec)

    theta(kk)=sum(Dtheta(1:kk)*Dt); 
    phi(kk)=sum(Dphi(1:kk)*Dt);
    fig.geometricPhase(kk)=sum(Dgeoph(1:kk)*Dt);
    fig.globalPhase(kk)=sum(Dgph(1:kk)*Dt);
    fig.dynamicPhase(kk)=sum(Ddyn(1:kk)*Dt);

end

%% Initialising angles - constant of integration
fig.thetaPhi(:,2) = phi - phi(1) + phi0;
fig.thetaPhi(:,1) = theta - theta(1) + theta0;