function Q = quaternion(quat)

Q=zeros(length(quat),4);

for kk=1:length(quat)

    Q(kk,1) = real( quat(1,1,kk) );
    Q(kk,2) = imag( quat(1,1,kk) );
    Q(kk,3) = real( quat(1,2,kk) );
    Q(kk,4) = imag( quat(1,2,kk) );
    
end