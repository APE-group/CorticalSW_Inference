%% Corresponding Author: Cristiano Capone, cristiano0capone@gmail.com
%  Paper: Simulations Approaching Data: Cortical Slow Waves in Inferred Models of the Whole Hemisphere of Mouse

%% Inner Loop

%%
close all
clear all

%%

for DATASETS = [1]
    
close all
clearvars -except DATASETS

% define data folder
nmouse = 1;
DataDir = 'SOData/';


%% define parameters

connectivityFree = 0;

slow_down_factor = 1.;

clear S

tauW = 0.1;
b0 = 40;
phi_init = pi/4;
phiMax = phi_init +.5;%pi/6+.05;
phiMin = phi_init -.5;%pi/6-.05;
  
TIME = 970;
dt = 0.04;
lambdaMax = 2.5*slow_down_factor;
lambdaMin = 1.*slow_down_factor;
lambda0 = 1.5*slow_down_factor;

k0Max = 15;
k0Min = 1;

ExcDriveMax = 3000;

spatialRegularization = 0;

%%


bFactor = ones(1,numel(DATASETS));
IExtFactor = ones(1,numel(DATASETS));

%%

count = 0;
S = [];

for nDataSet = DATASETS
%     

count = count+1;

%% load data

if nmouse ==1
DATA = load([DataDir 'initialized_images_t' num2str(nDataSet) '.mat']);
else
DATA = load([DataDir 'initialized_images_170111_t' num2str(nDataSet) '.mat']);
end

%%

A = DATA.images;

%%

clear isnanMat

SDATA{count} = zeros(2500,1000);
SDATA{count} = SDATA{count} + nan;
isnanMat = zeros(2500,1000);

find(1-isnan(sum(A)));
%%

SDATA{count} = A';

size(A)

SDATA{count}  = SDATA{count}(:,1:TIME);

S = [S SDATA{count}(:,1:TIME)];

end


%%

nonNANndx = find(1-isnan( sum(S(:,:),2) ));

%%

for count = 1:numel(DATASETS)
 
    SDATA{count} = SDATA{count}(nonNANndx,:);
end

%%

for k = 1:2500
    
    if nmouse==1

    x_pos(k) = floor((k-1)/50);
    y_pos(k) = mod(k-1,50);
    
    end
    
    if nmouse==2
        
    x_pos(k) = mod(k-1,50);
    y_pos(k) = floor((k-1)/50);
        
    end
   
end


%%

Map = NaN(50,50);

for k = 1:50*50

        IextMap(x_pos(k)+1,y_pos(k)+1) =  mean(S(k,:));

end

%%

Npop = numel(nonNANndx);

%%

x_pos_sel = x_pos(nonNANndx);
y_pos_sel = y_pos(nonNANndx);

clear dist

for i = 1:numel(nonNANndx)
    for j = 1:numel(nonNANndx)

        dist_(i,j) =  sqrt(( x_pos(nonNANndx(i))-x_pos(nonNANndx(j)))^2   +( y_pos(nonNANndx(i))-y_pos(nonNANndx(j)))^2  );
        
    end
end

%%

dist = sqrt((repmat(x_pos_sel,Npop,1) - repmat(x_pos_sel',1,Npop)).^2 + (repmat(y_pos_sel,Npop,1) - repmat(y_pos_sel',1,Npop)).^2 );

%% Deconvolution

for count = 1:numel(DATASETS)
  
Smat =  SDATA{count};
SConv =  SDATA{count};

TIME = size(SConv,2);

time = (1:TIME)*dt;

muLN = 2.2;
sigmaLN = 0.91;
filter = dt./time.*(1/sqrt(2*pi)/sigmaLN).*exp(-(log(time/(dt))-muLN).^2/(2*sigmaLN^2)) ;
filter(1) = 0;



for k = 1:size(Smat,1)

   
    X = smooth(Smat(k,:),1)';

    mins(k) = min(X);
    X = X - min(X);
 
    Y = fft(X);
    Ykernel = fft(filter);
    
    %%
     Fs = 1/dt;            % Sampling frequency
     L = TIME;             % Length of signal
     t = (0:L-1)*dt;        % Time vector
     
     %%
     
    f = Fs*(0:(L/2))/L;     
    ffs = [f fliplr(f)];
    
    ndx = find( ffs > 4 );% |  (ffs>3&ffs<=6) );%
    Y(ndx) = 0;
   
    Y_unfiltered = Y./Ykernel;
  
    %%
    X_unfiltered = real(ifft(Y_unfiltered));
    sDeconv{count}(k,:) = smooth(X_unfiltered,1);
   
end

end


%%

S = [];

for count = 1:numel(DATASETS)

    S = [S sDeconv{count}];
    
end

S = S.*heaviside(S);

%%

S = [];
for count = 1:numel(DATASETS)
S = [S sDeconv{count}];
end

maxS = (max(S(:)));

for count = 1:numel(DATASETS)

    S_ML = (sDeconv{count})/maxS;
    S_ML = S_ML.*heaviside(S_ML)*332.56;

    NuE_DATA{count} = S_ML;

end

%%
trPar = [5 5];
pot = 0;
nullClinesDrawing = 0; 
corrFact = 1;
selectionRule = 2;%3;
secondOrder = 0;

saveNum = 19;

%% Params Definition

potentiationLevel = 1.3;%1.25;%1.5;
depotentiationLevel = .8;
ExcDrive = 10;
popParam.b = [10 0]';

popParam.N = [1600 400]';
popParam.El = [-65 -65]';
popParam.Q = [1. -5.]'*5;

popParam.DeltaV = [2. .5]';

popParam.V_Reset = [-65 -65]';
popParam.tarp = [0.005 0.005]';
popParam.gl = [10 10]';
popParam.Cm = [.2 .2]';
popParam.theta = [-50 -50]';
popParam.taum = popParam.Cm./popParam.gl;

popParam.E = [0 -80]';

popParam.a = [0 0]';
popParam.tauW = [.4 .000001]';

popParam.tauSyn = [0.005 0.005]';

connParam.p = [0.25*potentiationLevel 0.25/2; 0.25*2 0.25*depotentiationLevel];
connParam.K = connParam.p.*repmat(popParam.N,1,numel(popParam.N))';

%%

spk_pot = -30;
Noise = 0;%2.5;
EXP = 1;
LIF = 1-EXP;
delay = 2*10^-3;%(ms)
delay_ndx = floor(delay/dt) ;

%%

eta = 1;

clear p

popParam.b = [b0*ones(Npop,1) 0*ones(Npop,1)]'/1;
popParam.tauW = [tauW .000001]';

k0 = 80 + randn(1,Npop)*0;
lambda = lambda0 + randn(1,Npop)*0;
gamma = 1./lambda;

a1 = .5 + randn(1,Npop)*0;
phi = phi_init + randn(1,Npop)*0;

a2 = -.6*0 + randn(1,Npop)*0;

DX = (repmat(x_pos_sel,Npop,1) - repmat(x_pos_sel',1,Npop));
DY = (repmat(y_pos_sel,Npop,1) - repmat(y_pos_sel',1,Npop));

rho = sqrt( DX.^2 + DY.^2 );
theta = atan(DY./DX);
theta(isnan(theta)) = 0;
theta( DX<0 ) = theta( DX<0 ) + pi ;

%%

phiRM = repmat(phi,Npop,1);
a1RM = repmat(a1,Npop,1);
a2RM = repmat(a2,Npop,1);

distEll = rho.*(1+a1RM.*cos(2*(theta+phiRM))).*( 1+a2RM.*cos(theta+phiRM) );

lambda = 1./gamma;

if 1-connectivityFree
    kLat = repmat(k0,Npop,1).*exp(-distEll./repmat(lambda,Npop,1))./repmat(lambda,Npop,1);
else
    kLat = repmat(k0,Npop,1).*exp(-distEll./repmat(lambda,Npop,1))./repmat(lambda,Npop,1)*0.01;%zeros(Npop,Npop);
end

ExcDrive = ones(size(NuE_DATA{1}(:,1)))*500;

%%

DeltaK=0;
etaK = 0.01;

Deltab = 0;
etab = 0.01;

DeltaNuext = 0;
etaNuext = .05;

Deltak0 = 0;
etak0 = .001;

DeltaPhi = 0;
etaPhi = .00001;

Deltaa1 = 0;
etaa1 = .000001;

Deltaa2 = 0;
etaa2 = .0001;

DeltaGamma = 0;
etaGamma = .00001;

%% Note- i have holy NuE data, so I have to Guess the others

for count = 1:numel(DATASETS)
    W_DATA{count} = zeros(Npop,1);
    for t = 1:TIME-1
        W_DATA{count}(:,t+1) = W_DATA{count}(:,t) + dt.*( - W_DATA{count}(:,t)./popParam.tauW(1)  + NuE_DATA{count}(:,t) );
    end
end

%% separate maximization data and validation data

TIME = floor(TIME*.8);

%%
for count = 1:numel(DATASETS)

NuE_VAL{count} = NuE_DATA{count}(:,TIME+1:end);
NuE_DATA{count} = NuE_DATA{count}(:,1:TIME);

W_VAL{count}= W_DATA{count}(:,TIME+1:end);
W_DATA{count} = W_DATA{count}(:,1:TIME);

end

%%

eta = 0.001;

p = 0;

for dataSetNum = 1:numel(DATASETS)
    
    [TFEdMu,TFE] = tfFunctionDerParallel0(NuE_DATA{dataSetNum},W_DATA{dataSetNum}.*repmat(popParam.b(1,:)' *bFactor(dataSetNum),1,TIME), popParam, ExcDrive*IExtFactor(dataSetNum), kLat);
    p = p +  -sum(sum((NuE_DATA{dataSetNum}(:,2:end)-TFE(:,1:end-1)).^2));
    
    likelihood = p;
    
end

%%

figure(1)
clf

%%

plotEvery = 50;
clear llk llkVal
saveVideoPars = 0;
maxMethod = 1;

if saveVideoPars
    v = VideoWriter('ParametersOptimization.avi','MPEG-4');
    v.Quality = 80;
    open(v);
end

count = 0;    

%% inner loop 

for iter = 1:701%1200

  count = count+1;

  for dataSetNum = 1:numel(DATASETS)

  [TFEdMu_{dataSetNum},TFE_{dataSetNum}] = tfFunctionDerParallel0(NuE_DATA{dataSetNum},W_DATA{dataSetNum}.*repmat(popParam.b(1,:)',1,TIME) *bFactor(dataSetNum),popParam,ExcDrive*IExtFactor(dataSetNum),kLat);
    
  dLdMu{dataSetNum} = 2*((NuE_DATA{dataSetNum}(:,2:end) - TFE_{dataSetNum}(:,1:end-1)).* TFEdMu_{dataSetNum}(:,1:end-1) );

  end
 
  
  %% save old ones
  
  DeltabOld = Deltab;
  Deltak0Old = Deltak0;
  DeltaNuextOld = DeltaNuext;
  DeltaGammaOld = DeltaGamma;
  Deltaa1Old = Deltaa1;
  Deltaa2Old = Deltaa2;
  
  DeltaPhiOld = DeltaPhi;
  
  %%
   
  Deltak0 = 0;%Deltak0+ sum(dLdMu{dataSetNum}*dMudK{dataSetNum}'.*exp(-distEll./repmat(lambda,Npop,1)),1);
  DeltaGamma = 0;%DeltaGamma+ sum( dLdMu{dataSetNum}*dMudK{dataSetNum}'.*kLat.*(1-distEll.*(repmat(gamma,Npop,1)).^2),1);

  Deltaa1 = 0;%Deltaa1+ sum( dLdMu{dataSetNum}*dMudK{dataSetNum}'.*kLat./(-repmat(lambda,Npop,1)).* dd_da1 , 1 );
  Deltaa2 = 0;%Deltaa2+ sum( dLdMu{dataSetNum}*dMudK{dataSetNum}'.*kLat./(-repmat(lambda,Npop,1)).* dd_da2 , 1 );
  
  DeltaPhi = 0;%DeltaPhi+ sum( dLdMu{dataSetNum}*dMudK{dataSetNum}'.*kLat./(-repmat(lambda,Npop,1)).*dd_dphi  , 1 );
  DeltaNuext = 0;  
  Deltab = 0;
  
  %% DeltaK
  
  if connectivityFree
      
      %%
      
      clear dMudK
for dataSetNum = 1:numel(DATASETS)
  
  dMudK{dataSetNum} = NuE_DATA{dataSetNum}(:,1:end-1)*popParam.tauSyn(1)*popParam.Q(1); 
  
  DeltaKOld = DeltaK;
  DeltaK = dLdMu{dataSetNum}*dMudK{dataSetNum}';
  
  epsK = sign(DeltaKOld).*sign(DeltaK);
  
  etaK = etaK.*0.999.^-epsK;
  
  
  kLat_0 = kLat;
  kLat = kLat + etaK.*sign(DeltaK);
  kLat = max(kLat,0);
  
end

%%
  end
  
  %%

  for dataSetNum = 1:numel(DATASETS)
   
      
  %% Delta b K 
  
  dMudb{dataSetNum} = - W_DATA{dataSetNum}(:,1:end-1)*bFactor(dataSetNum);   
  
  dMudbFact{dataSetNum} = - W_DATA{dataSetNum}(:,1:end-1).*repmat(popParam.b(1,:)',1,TIME-1);
  
  DeltabFact(dataSetNum) = sum( sum(dLdMu{dataSetNum}.*dMudbFact{dataSetNum} ) );
  
  Deltab = Deltab + diag( dLdMu{dataSetNum}*dMudb{dataSetNum}'  );
  
  dMudK{dataSetNum} = NuE_DATA{dataSetNum}(:,1:end-1)*popParam.tauSyn(1)*popParam.Q(1); 
  
  dMuNuext{dataSetNum} = (popParam.tauSyn(1)*popParam.Q(1))*IExtFactor(dataSetNum);
  
  dMuNuextFact{dataSetNum} = (popParam.tauSyn(1)*popParam.Q(1))*NuE_DATA{dataSetNum}(:,1:end-1);

  DeltaIExtFactor(dataSetNum) = sum(sum(dMuNuextFact{dataSetNum} .*dMuNuext{dataSetNum}));
  
  DeltaNuext = DeltaNuext + sum( dLdMu{dataSetNum}*dMuNuext{dataSetNum}' ,2);
  
 %% delta k0 , lambda0 a, phi
 
 
  phiRM = repmat(phi,Npop,1);
  a1RM = repmat(a1,Npop,1);
  a2RM = repmat(a2,Npop,1);

  distEll = rho.*(1+a1RM.*cos(2*(theta+phiRM))).*( 1 + a2RM.*cos(theta+phiRM) );

  Deltak0 =Deltak0+ sum(dLdMu{dataSetNum}*dMudK{dataSetNum}'.*exp(-distEll./repmat(lambda,Npop,1)),1);
  DeltaGamma =DeltaGamma+ sum( dLdMu{dataSetNum}*dMudK{dataSetNum}'.*kLat.*(1-distEll.*(repmat(gamma,Npop,1)).^2),1);

  dd_da1 = rho.*cos(2*(theta+phiRM)).*( 1+a2RM.*cos(theta+phiRM));
  dd_da2 = rho.*(1+a1RM.*cos(2*(theta+phiRM))) .*cos(theta+phiRM) ;
  
  dd_dphi = -rho.*( 2*sin(2*(theta+phiRM)).*a1RM.*( 1+a2RM.*cos(theta+phiRM) )  +  (1+a1RM.*cos(2*(theta+phiRM))).*a2RM.*sin(theta+phiRM)      )  ;
  
  Deltaa1 =Deltaa1+ sum( dLdMu{dataSetNum}*dMudK{dataSetNum}'.*kLat./(-repmat(lambda,Npop,1)).* dd_da1 , 1 );
  Deltaa2 =Deltaa2+ sum( dLdMu{dataSetNum}*dMudK{dataSetNum}'.*kLat./(-repmat(lambda,Npop,1)).* dd_da2 , 1 );
  
  DeltaPhi =DeltaPhi+ sum( dLdMu{dataSetNum}*dMudK{dataSetNum}'.*kLat./(-repmat(lambda,Npop,1)).*dd_dphi  , 1 );
      
  end
  
  %%
  
  bFactor = bFactor + 10^-10*DeltabFact;
  bFactor = bFactor/bFactor(1);
  IExtFactor = IExtFactor + 2*10^-7*DeltaIExtFactor;
  IExtFactor = IExtFactor/IExtFactor(1);
  
  IExtFactor = min(IExtFactor,3);
  
  bFactor
  IExtFactor
  
  %% eps e eta updates
  
  delta = 0.99;
  epsb = sign(DeltabOld).*sign(Deltab);
  etab = etab.*delta.^-epsb;

  epsNuext = sign(DeltaNuextOld).*sign(DeltaNuext);
  etaNuext = etaNuext.*delta.^-epsNuext;
 
  epsk0 = sign(Deltak0Old).*sign(Deltak0);
  etak0 = etak0.*delta.^-epsk0;
  
  epsGamma = sign(DeltaGammaOld).*sign(DeltaGamma);
  etaGamma = etaGamma.*delta.^-epsGamma;
  
  epsa1 = sign(Deltaa1Old).*sign(Deltaa1);
  etaa1 = etaa1.*delta.^-epsa1;
  
  epsa2 = sign(Deltaa2Old).*sign(Deltaa2);
  etaa2 = etaa2.*delta.^-epsa2;

  epsPhi = sign(DeltaPhiOld).*sign(DeltaPhi);
  etaPhi = etaPhi.*delta.^-epsPhi;
  
%% save old weights

  ExcDrive_0 = ExcDrive;
  b_0 = popParam.b(1,:) ;
  k0_0 = k0;
  gamma_0 = gamma;
  a1_0 = a1;
  a2_0 = a2;
  
  phi_0 = phi;
  
%% updating weights irProp


 if maxMethod==1
 
     
  ExcDrive = max(ExcDrive + etaNuext.*sign(DeltaNuext),100);
   ExcDrive = min(ExcDrive,ExcDriveMax);
  
  popParam.b(1,:) =  max(popParam.b(1,:)+(etab.*sign(Deltab))',0);
  k0 = max(k0 + etak0.*sign(Deltak0),k0Min);
  k0 = min(k0,k0Max);
  gamma = gamma + etaGamma.*sign(DeltaGamma);
  
  gamma = max(gamma,1/lambdaMax);
  gamma = min(gamma,1/lambdaMin);
  
  a1 =  a1 + etaa1.*sign(Deltaa1);
  a2 =  a2 + etaa2.*sign(Deltaa2);
  
  phi =  phi + etaPhi.*sign(DeltaPhi);
  
  K = floor(phi/(2*pi));
  phi = phi - 2*K*pi;

  phi = min(phi,phiMax);
  phi = max(phi,phiMin);
  
  a1 = min(a1,0.6);
  a1 = max(a1,.05);
  
  a2 = min(a2,0.5);
  a2 = max(a2,-0.5);
  
   %% spatial regularization
 
  lamReg = .05;
 
  disReg = exp(-rho/lamReg);
  disReg = disReg./repmat( sum(disReg,2) , 1, Npop  );
  
   if spatialRegularization
  
  gamma = gamma*disReg;
  phi = phi*disReg;
  a1 = a1*disReg;
  a2 = a2*disReg;
  k0 = k0*disReg;
  ExcDrive = disReg*ExcDrive;
  popParam.b(1,:) = popParam.b(1,:)*disReg;
  
   end
 end
  
 %%
 
 if maxMethod==1
     
  phiRM = repmat(phi,Npop,1);
  a1RM = repmat(a1,Npop,1);
  a2RM = repmat(a2,Npop,1);

  distEll = rho.*(1+a1RM.*cos(2*(theta+phiRM))).*( 1+a2RM.*cos(theta+phiRM) );
  
  lambda = 1./gamma;
  
  if 1-connectivityFree
    kLat = repmat(k0,Npop,1).*exp(-distEll./repmat(lambda,Npop,1))./repmat(lambda,Npop,1);
  end

  p = 0;
  
%  bFactor(2) = bFactor(2)*1.1;
  
  for dataSetNum = 1:numel(DATASETS)
  
   [TFEdMu,TFE] = tfFunctionDerParallel0(NuE_DATA{dataSetNum},W_DATA{dataSetNum}.*repmat(popParam.b(1,:)' *bFactor(dataSetNum),1,TIME), popParam, ExcDrive*IExtFactor(dataSetNum), kLat);
   p = p +  -sum(sum((NuE_DATA{dataSetNum}(:,2:end)-TFE(:,1:end-1)).^2));
   
   likelihood_0 = likelihood;
   likelihood = p;
   
  end
  
  
if likelihood<likelihood_0  
    
     if connectivityFree
         
     
    
     kLat(epsK(:)<0) =  kLat_0(epsK(:)<0);
     DeltaK(epsK(:)<0)=0;
     end

     ExcDrive(epsNuext(:)<0) =  ExcDrive_0(epsNuext(:)<0);
     popParam.b(1,epsb(:)<0) =  b_0(epsb(:)<0);
     k0(epsk0(:)<0) =  k0_0(epsk0(:)<0);
     a1(epsa1(:)<0) =  a1_0(epsa1(:)<0);
     a2(epsa2(:)<0) =  a2_0(epsa2(:)<0);
     
     gamma(epsGamma(:)<0) =  gamma_0(epsGamma(:)<0);
     phi(epsPhi(:)<0) =  phi_0(epsPhi(:)<0);
    
     likelihood = likelihood_0;
    
end

  phiRM = repmat(phi,Npop,1);
  a1RM = repmat(a1,Npop,1);
  a2RM = repmat(a2,Npop,1);

  distEll = rho.*(1+a1RM.*cos(2*(theta+phiRM))).*( 1+a2RM.*cos(theta+phiRM) );
  
  lambda = 1./gamma;
  
  if 1-connectivityFree
  kLat = repmat(k0,Npop,1).*exp(-distEll./repmat(lambda,Npop,1))./repmat(lambda,Npop,1);
  end
 
end
 

%%

pVal = 0;

 for dataSetNum = 1:numel(DATASETS)

   [TFEdMu,TFEVal] = tfFunctionDerParallel0(NuE_VAL{dataSetNum},W_VAL{dataSetNum}.*repmat(popParam.b(1,:)' *bFactor(dataSetNum),1,size(NuE_VAL{1},2)) ,popParam,ExcDrive*IExtFactor(dataSetNum),kLat);
 
    pVal = pVal - sum(sum((NuE_VAL{dataSetNum}(:,2:end)-TFEVal(:,1:end-1)).^2));

    llk(count) = likelihood/TIME/Npop;
    llkVal(count) = pVal/200/Npop;
    
 end
   
%% Plot a subset of parameters

    if mod(count,plotEvery)==1
    
    figure(1)
    
    clf
 
    %%
    
    subplot(3,3,4)

     IextMap = NaN(50,50);

    for k = 1:Npop

        IextMap(x_pos_sel(k)+1,y_pos_sel(k)+1) =  ExcDrive(k);

    end

    imagesc(IextMap)

    title('I_{ext}: external current')
    colorbar()
    set(gca,'clim',[ -10 max(IextMap(:))+100])
    
    %%
    
     a2Map = NaN(50,50);

    for k = 1:Npop
       a2Map(x_pos_sel(k)+1,y_pos_sel(k)+1) =  a2(k);
    end

    
    %%
    
    subplot(3,3,1)
    
    k0Map = NaN(50,50);

    for k = 1:Npop

        k0Map(x_pos_sel(k)+1,y_pos_sel(k)+1) =  k0(k);

    end

    imagesc(k0Map)
    set(gca,'clim',[-1 max(k0Map(:))])
    title('k_0')
    colorbar()
   
    
   % else
 subplot(3,3,8)
scatter(dist(:),kLat(:),'b.')
hold on

for d = 0:40
    
    ndx_d = find(dist(:)>=d & dist(:)<d+1  );
    kDecay(d+1) = mean(kLat(ndx_d));
    stdErr(d+1) = std(kLat(ndx_d))/numel((ndx_d));
    distance(d+1) = d;
    
end

[pp]=polyfit(distance(1:15),log(kDecay(1:15))./distance(1:15),1);
lambda1 = -1/pp(1);

[pp]=polyfit(distance(2:15),log(kDecay(2:15)./distance(2:15)),1);

lambda2 = -1/pp(1);

errorbar(distance,kDecay,stdErr,'r-','linewidth',2.5)
title(lambda2)

    %%
    
    subplot(3,3,7)
    
    lambdaMap = NaN(50,50);

    for k = 1:Npop
        lambdaMap(x_pos_sel(k)+1,y_pos_sel(k)+1) =  lambda(k);
    end

    imagesc(lambdaMap*100)
    title('\lambda (\mu m)')
    colorbar()
    set(gca,'dataaspectratio',[1 1 1])

    %%
    
    subplot(3,3,5)

    plot(popParam.b(1,:),'o')

    bMap = NaN(50,50);

    for k = 1:Npop

        bMap(x_pos_sel(k)+1,y_pos_sel(k)+1) =  popParam.b(1,k);

    end

    imagesc(bMap)
    title('b: adaptation')
    
    set(gca,'clim',[ 0 max(bMap(:))])
    colorbar()
    
    %%

    subplot(3,3,6)
    hold on
    plot(-llk)
    
    xlabel('#iterations')
    ylabel('likelihood')
    set(gca,'yscale','log')
    
    %%
    
     subplot(3,3,9)
    hold on
    plot(-llkVal,'r')
    
    xlabel('#iterations')
    ylabel('validation likelihood')
    set(gca,'yscale','log')
    
    %%
    
    subplot(3,3,2)

      phiMap = NaN(50,50);
            
for k = 1:Npop
    
    phiMap(x_pos_sel(k)+1,y_pos_sel(k)+1) =  phi(k);
    
end

    imagesc(phiMap)
    title('\phi: orientation')
    
    set(gca,'clim',[ 0 max(phiMap(:))])
    colorbar()

    %%
    
    subplot(3,3,3)
  
    aMap = NaN(50,50);

    for k = 1:Npop

        aMap(x_pos_sel(k)+1,y_pos_sel(k)+1) =  a1(k);

    end

    imagesc(aMap)

    title('e: eccentricity')
    colorbar()
    
    set(gca,'clim',[ 0 max(aMap(:))])

    %%
    
    set(gcf, 'PaperUnits', 'inch', 'PaperPosition', [0 0 8 3]);
    set(gcf,'color','w');
    
  cm = parula(1000);
  cm = [[1 1 1]; cm ];

  colormap(cm)

    drawnow()
    
    if saveVideoPars
        
      frame = getframe(gcf);
      writeVideo(v,frame);
    end

    end

%%

if mod(count,100)==1

    %%
popParam.tauW(1) = tauW;
b_fact = 1.;
kFact = 1.; 

dt = 0.04;

NuE = abs(randn(Npop,1)*10);%[1 ; 2];
NuI = zeros(Npop,1);
W = zeros(Npop,1);
 
driven = 0;
 
PERIOD = 2.5;
AMP0 = 1;
D_AMP = .8;

for t = 1:TIME-1
    
    AMP = (AMP0 + D_AMP*cos(2*pi*dt*t/PERIOD) ) /(AMP0+D_AMP  );
    
    if t == TIME/2
        driven = 0;
    end
    
    if driven
        NuE_inp(:,t) =  NuE_DATA{1}(:,t);
    else
        NuE_inp(:,t) =  NuE(:,t);
    end
   
    [dersSim,TFESim] = tfFunctionDerParallel0(NuE_inp(:,t),popParam.b(1,:)'.*bFactor(1).*W(:,t),popParam, ExcDrive*AMP*IExtFactor(1) , kLat *kFact);%1.03
    
    NuE(:,t+1) =  randn(Npop,1).*(sqrt(abs(TFESim)/2/popParam.N(1)/dt)*2 + 12 ) + TFESim;% + NoiseLevel*randn(1,2)'.*sqrt((Nu(:,t) + 1 )./popParam.N)*sqrt(dt);
  
    
    NuE(:,t+1) = max(NuE(:,t+1),0);
     
    W(:,t+1) = W(:,t) + dt.*( - W(:,t)./popParam.tauW(1)  + NuE_inp(:,t) );
    
end

for k = 1:Npop
     NuE(k,:) = smooth(NuE(k,:),5);
end

%%
% FFTs
Fs = 1/dt;            % Sampling frequency                    
% dt = 1/Fs;             % Sampling period       
L = TIME;             % Length of signal
t = (0:L-1)*dt;        % Time vector
f = Fs*(0:(L/2))/L;
f_data = f;

FFT_SIM = fft(mean(NuE,1));
FFT_SIM = abs(FFT_SIM/L);
FFT_SIM = FFT_SIM(1:L/2+1);
FFT_SIM(2:end-1) = 2*FFT_SIM(2:end-1);

%%

FFT_DATA = real(fft(mean(NuE_DATA{1},1)));

FFT_DATA = abs(FFT_DATA/L);
FFT_DATA = FFT_DATA(1:L/2+1);
FFT_DATA(2:end-1) = 2*FFT_DATA(2:end-1);

%%

figure(2)
clf
subplot(4,1,1)
imagesc(NuE)
title('S_e inferred')
colorbar
set(gca,'clim',[0 120])
subplot(4,1,2)

imagesc(NuE_DATA{1})
title('S_e data')
colorbar
set(gca,'clim',[0 120])

subplot(4,1,3)

plot(mean(NuE))
hold on
plot(mean(NuE_DATA{1}))
colorbar

subplot(4,1,4)
hold on
plot(f,smooth(FFT_SIM))
plot(f,smooth(FFT_DATA))
set(gca,'yscale','log','xscale','log')
legend('sim','data')

colormap(parula)
drawnow()

end

%%

end

%%
save(['workSpace_t' num2str(DATASETS) '_mouse' num2str(nmouse) '_phi_pi4_freephi_lam0' num2str(lambda0) '.mat'])


end

%%

figure()
k_example = NaN(50,50);

for subp = 1:10
    
    subplot(2,5,subp)
    
    ndx=randperm(Npop);
    ndx=ndx(1);

 for k = 1:Npop

        k_example(x_pos_sel(k)+1,y_pos_sel(k)+1) =  kLat(k,ndx);

 end
 
 
hold on
imagesc(k_example')
plot(x_pos_sel(ndx),y_pos_sel(ndx),'ro')
set(gca,'clim',[-1 1],'ylim',[0 50],'xlim',[0 50])
title('k_0')
set(gca,'ydir','reverse')
colormap(copper)
    

end

    