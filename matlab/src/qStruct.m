%% Define the quaternion structure 
function q = qStruct(cay)

q.time  = linspace(0,2*pi,2^11)';
q.thetaPhi(:,1) = linspace(0+0.005,pi-0.005,1000);
q.thetaPhi(:,2) = linspace(0+0.005,2*pi-0.005,1000);

%% Initialize
q.fermion = zeros(2,2,length(q.time));
q.boson = zeros(2,2,length(q.time));
q.mixed = zeros(2,2,length(q.time));
q.spinhalf = zeros(2,2,length(q.time));
q.figure1 = zeros(2,2,length(q.time));

%% Define the unitaries
for kk=1:length(q.time)
q.fermion(:,:,kk) = expm(-cay.I*q.time(kk))*expm(cay.K*q.time(kk)/2)*expm(cay.I*q.time(kk));
q.boson(:,:,kk) = expm(-cay.I*q.time(kk)/2)*expm(-cay.J*q.time(kk))*expm(-cay.I*q.time(kk));
q.mixed(:,:,kk) = expm(-cay.I*q.time(kk))*expm(cay.K*q.time(kk)/2)*expm(cay.I*q.time(kk)/2)*expm(-cay.J*q.time(kk))*expm(-cay.I*q.time(kk));
q.spinhalf(:,:,kk) = expm(cay.I*q.time(kk)/2)*expm(cay.J*q.time(kk)/2);
q.figure1(:,:,kk) = expm( -cay.I*q.time(kk)/2 )*expm( -cay.K*q.time(kk)/2 )*expm( cay.I*q.time(kk)/2 ); 
end
