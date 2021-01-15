%function MAIN
addpath('src')  
%% The cayley matrices
cay = cayleyMatrices;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Quaternions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q = qStruct(cay);
fprintf('Generating the quaternion data .... \n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Unitaries %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Defining The Path Generator Quaternion
%% Call the quaternion
boson.unitary = quaternion(q.boson);
%% Call the quaternion
fermion.unitary = quaternion(q.fermion);
%% Call the quaternion
mixed.unitary = quaternion(q.mixed);
%% Call the quaternion
spinhalf.unitary = quaternion(q.spinhalf);
%% Call the quaternion
figure1.unitary = quaternion( q.figure1 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figure #1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure1 = bloch( figure1.unitary, q.time, q.thetaPhi(180,1), q.thetaPhi(400,2) );
%% The Darboux Frames
darboux1 = movingFrames( figure1.blochVector, figure1.thetaPhi, q.time );
%% The tangent vector
figure1.tangentVector = parallelTransport(  -[-1/sqrt(2);-1/sqrt(2)], darboux1.surfaceFrame, figure1.geometricPhase, q.time);
%% Plot figure 1
fprintf('Displaying Figure #1 .... \n');
makeFigure1(darboux1,figure1,'Figure #1')
askContinue('Press enter to generate the 2 sphere plot; figure #4(a)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fermions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Call the quaternion
fermion.unitary = quaternion(q.fermion);
fermionA = bloch( fermion.unitary, q.time, q.thetaPhi(750,1), q.thetaPhi(1,2) );
fermionB = bloch( fermion.unitary, q.time, q.thetaPhi(500,1), q.thetaPhi(500,2) );
fermionC = bloch( fermion.unitary, q.time, q.thetaPhi(100,1), q.thetaPhi(1,2) );
%% The Darboux Frames
darbouxFermionA = movingFrames( fermionA.blochVector, fermionA.thetaPhi, q.time );
darbouxFermionB = movingFrames( fermionB.blochVector, fermionB.thetaPhi, q.time );
darbouxFermionC = movingFrames( fermionC.blochVector, fermionC.thetaPhi, q.time );
%% The tangent vector
fermionA.tangentVector = parallelTransport(  [1/sqrt(2);1/sqrt(2)], darbouxFermionA.surfaceFrame, fermionA.geometricPhase, q.time);
fermionB.tangentVector = parallelTransport(  [1/sqrt(2);1/sqrt(2)], darbouxFermionB.surfaceFrame, fermionB.geometricPhase, q.time);
fermionC.tangentVector = parallelTransport(  [1/sqrt(2);1/sqrt(2)], darbouxFermionC.surfaceFrame, fermionC.geometricPhase, q.time);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figure #4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot figure 4 - top row
fprintf('Displaying Figure #4(a) - Parallel transport and the geometric phase .... \n')
make2sphere(darbouxFermionA,fermionA,'Figure 4(a) - the geometric phase')
askContinue('Press enter to generate figure #4(b)')
fprintf('Displaying Figure #4(b) - Parallel transport and the geometric phase .... \n')
make2sphere(darbouxFermionB,fermionB,'Figure 4(b) - the geometric phase')
askContinue('Press enter to generate figure #4(c)')
fprintf('Displaying Figure #4(c) - Parallel transport and the geometric phase .... \n')
make2sphere(darbouxFermionC,fermionC,'Figure 4(c) - the geometric phase')
askContinue('Press enter to generate figure #4(a)')

%% Plot figure 4 - middle and bottom rows
fprintf('Displaying figure #4(a) - The S1 fiber bundle .... \n')
makeFiberPlot(fermionA,q.time,'Figure 4(a) - The S1 fiber bundle')
askContinue('Press enter to generate figure #4(b)')
fprintf('Displaying figure #4(b) - The S1 fiber bundle .... \n')
makeFiberPlot(fermionB,q.time,'Figure 4(b) - The S1 fiber bundle')
askContinue('Press enter to generate figure #4(c)')
fprintf('Displaying figure #4(c) - The S1 fiber bundle .... \n')
makeFiberPlot(fermionC,q.time,'Figure 4(c) - The S1 fiber bundle')
askContinue('Press enter to generate figure #5(a)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bosons %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Call the quaternion
boson.unitary = quaternion(q.boson);
bosonA = bloch( boson.unitary, q.time, q.thetaPhi(750,1), q.thetaPhi(750,2) );
bosonB = bloch( boson.unitary, q.time, q.thetaPhi(500,1), q.thetaPhi(500,2) );
bosonC = bloch( boson.unitary, q.time, q.thetaPhi(250,1), q.thetaPhi(250,2) );
%% The Darboux Frames
darbouxBosonA = movingFrames( bosonA.blochVector, bosonA.thetaPhi, q.time );
darbouxBosonB = movingFrames( bosonB.blochVector, bosonB.thetaPhi, q.time );
darbouxBosonC = movingFrames( bosonC.blochVector, bosonC.thetaPhi, q.time );
%% The tangent vector
bosonA.tangentVector = parallelTransport(  [1/sqrt(2);1/sqrt(2)], darbouxBosonA.surfaceFrame, bosonA.geometricPhase, q.time);
bosonB.tangentVector = parallelTransport(  [1/sqrt(2);1/sqrt(2)], darbouxBosonB.surfaceFrame, bosonB.geometricPhase, q.time);
bosonC.tangentVector = parallelTransport(  [1/sqrt(2);1/sqrt(2)], darbouxBosonC.surfaceFrame, bosonC.geometricPhase, q.time);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figure #5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plot figure 5 - top row
fprintf('Displaying figure #5(a) - Parallel transport and the geometric phase .... \n')
make2sphere(darbouxBosonA,bosonA,'Figure 5(a) - the geometric phase')
askContinue('Press enter to generate figure #5(b)')
fprintf('Displaying figure #5(b) - Parallel transport and the geometric phase .... \n')
make2sphere(darbouxBosonB,bosonB,'Figure 5(b) - the geometric phase')
askContinue('Press enter to generate figure #5(c)')
fprintf('Displaying figure #5(c) - Parallel transport and the geometric phase .... \n')
make2sphere(darbouxBosonC,bosonC,'Figure 5(c) - the geometric phase')
askContinue('Press enter to generate figure #5(a)')

%% Plot figure 5 - middle and bottom rows
fprintf('Displaying figure #5(a) - The S1 fiber bundle .... \n')
makeFiberPlot(bosonA,q.time,'Figure 5(a) - The S1 fiber bundle')
askContinue('Press enter to generate figure #5(b)')
fprintf('Displaying figure #5(b) - The S1 fiber bundle .... \n')
makeFiberPlot(bosonB,q.time,'Figure 5(b) - The S1 fiber bundle')
askContinue('Press enter to generate figure #5(c)')
fprintf('Displaying figure #5(c) - The S1 fiber bundle .... \n')
makeFiberPlot(bosonC,q.time,'Figure 5(c) - The S1 fiber bundle')
askContinue('Press enter to generate figure #B1 - the darboux frames')

%%%%%%%%%%%%% Darboux surface - Darboux curve - Frenet Serret %%%%%%%%%%%%%

fprintf('Displaying the darboux surface frame .... \n')
makeDarboux(darbouxFermionA.surfaceFrame,fermionA,'Darboux Surface Frame')
askContinue('Press enter to generate the darboux curve frame')
fprintf('Displaying the darboux curve frame .... \n')
makeDarboux(darbouxFermionB.curveFrame,fermionB,'Darboux Curve Frame')
askContinue('Press enter to generate the frenet serret frame')
fprintf('Displaying the frenet serret frame .... \n')
makeDarboux(darbouxBosonB.frenetSerretFrame,bosonB,'Frenet Serret Frame')
askContinue('Press enter for the global phase of the closed path')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figure #2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figure #3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figure #6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% S1 fiber bundle of the closed paths
closedPaths = bundleSpace( q, fermion.unitary, boson.unitary, mixed.unitary, spinhalf.unitary, 20); 
%The last entry is the resolution and generates nxn matrices for the fiber bundle of the closed path. 
%Low resolution; n = 10 -> High resolution; n = 100+

%% Plot figures 2 and 3
closedPathFigure(closedPaths,'Figure #2: Global Phase of the closed path (a) fermion (b) boson.','Figure #3: Global and dynamic phase of the closed path (a) mixed global phase (b) mixed dynamic phase.','Figure #6: Spin half particles and the Stern Gerlach experiment.')
