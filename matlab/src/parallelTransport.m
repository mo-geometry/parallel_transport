function TanVec = parallelTransport(state,DS,geoPh,tvec)

%% The time step and initial state
Dt=tvec(2)-tvec(1);
Vth0=state(1);Vph0=state(2);

DSop=zeros(2,2,length(tvec));
Rotor=zeros(2,2,length(tvec));

%% Extracting the frame
eTh = DS(:,4:6); eP = DS(:,7:9);

%% The derivatives
DeTh = derivative(eTh,Dt);  DeP = derivative(eP,Dt);

%% The differential forms
LThTh = sum(DeTh.*eTh,2);	LPTh = sum(DeTh.*eP,2);
LThP = sum(DeP.*eTh,2); 	LPP = sum(DeP.*eP,2);

%% Creating the exponential operator

for kk=1:length(tvec)
    DSop(1:2,1:2,kk)=[LThTh(kk),LPTh(kk);LThP(kk),LPP(kk)];
    Rotor(1:2,1:2,kk)=[cos(geoPh(kk)),-sin(geoPh(kk));sin(geoPh(kk)),cos(geoPh(kk))];
end

%% Numerical integration of the tangent vector
TanVec=zeros(2,length(tvec));
TanVec(:,1)=[Vth0;Vph0];

for kk=1:length(tvec)-1
%     TanVec(:,kk+1)=expm(DSop(:,:,kk)*Dt)*TanVec(:,kk);
    TanVec(:,kk+1)=Rotor(1:2,1:2,kk)'*TanVec(:,1);
end

TanVec=TanVec';

function De1 = derivative(e1,Dt)
De1i =  gradient(e1(:,1),Dt);
De1j =  gradient(e1(:,2),Dt);
De1k =  gradient(e1(:,3),Dt);
De1 = [De1i De1j De1k];