function  [TFEdMu,TFE] = tfFunctionDerParallel0(NuE,W,popParam,ExcDrive,kLat)%,stimulus)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


    tFun = @(x) 1./(popParam.tarp(1) - popParam.taum(1).*log( max( ((x.*popParam.taum(1)-(popParam.theta(1) - popParam.El(1)))./(x.*popParam.taum(1)-popParam.V_Reset(1))),0.00000000000001) )  ) ; 

    ExcDrive = repmat(ExcDrive,1,size(NuE,2));

    i = 1;
    j=1; muGE = kLat*(NuE) + ExcDrive; %from j = 1

    %% Theoretical estimation for MuV
    
    %stim = popParam.weights_stim * popParam.stimulus_temporal_profile;
    
    muE = ( muGE - W);% + stimulus );
    eps = 1.;
    
    %TFtmp = TF_ADEX_Sampled_N( muE(:) , sqrt(sigma2E(:))*sqrt(popParam.tauSyn(1))/sqrt(2) ,muI(:), sqrt(sigma2I(:))*sqrt(popParam.tauSyn(2))/sqrt(2));
    %TFtmppEps = TF_ADEX_Sampled_N( muE(:) + eps , sqrt(sigma2E(:))*sqrt(popParam.tauSyn(1))/sqrt(2) ,muI(:), sqrt(sigma2I(:))*sqrt(popParam.tauSyn(2))/sqrt(2));

   TFtmp = TF_ADEX_Sampled_N( muE(:) , 0 , 0, 0 );
   TFtmppEps = TF_ADEX_Sampled_N( muE(:) + eps , 0,0,0);
%    
%     TFtmp = tFun(muE(:));     
%     TFtmppEps =  tFun(muE(:)+eps);
  
    TFE = zeros(size(muE));
    TFEdMu = zeros(size(muE));
    TFEdSig2 = 0;
     
    TFEdMu(:) = (TFtmppEps(1:numel(muE))-TFtmp(1:numel(muE)))/eps;   
    TFEdMu = max(TFEdMu,0.0);
     
    TFE = zeros(size(muE));
    TFE(:) = TFtmp(1:numel(muE));
    
    TFE = max(TFE,0.00);
     

end

