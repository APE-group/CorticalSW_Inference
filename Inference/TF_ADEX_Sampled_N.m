function [  nus ] = TF_ADEX_Sampled_N( mue, sigmae, mui , sigmai )
% 

% load('sfS.mat','sfRS','sfFS')
% 
% % p = 1;
% % tp = tpaps([muS(:) sigS(:)]',TF_RS_Sampled(:)',p);
% % tpFS = tpaps([muS(:) sigS(:)]',TF_FS_Sampled(:)',p);
% 
% %nuRS = fnval(tpRS,[mue'; sigmae']);
% nuRS = sfRS(mue', sigmae');
% % sigmae
% nuRS(mue <-1000) = 0;
% 
% % nuRS(mue <400) = 0;
% 
% 
% nuFS = sfFS(mui', sigmai');
% %nuFS = fnval(tpFS,[mui'; sigmai']);
% nuFS(mui <400) = 0;
% 
% 
% nuRS = min(nuRS,200);
% nuFS = min(nuFS,200);
% 
% nuRS = max(nuRS,0);
% nuFS = max(nuFS,0);
% 
% nus = [nuRS nuFS];

%%

% 
% load('TF_tps.mat','tpRS','tpFS')
% 
% % p = 1;
% % tp = tpaps([muS(:) sigS(:)]',TF_RS_Sampled(:)',p);
% % tpFS = tpaps([muS(:) sigS(:)]',TF_FS_Sampled(:)',p);
% 
% nuRS = fnval(tpRS,[mue'; sigmae']);
% 
% nuRS(mue <-400) = 0;
% 
% 
% nuFS = fnval(tpFS,[mui'; sigmai']);
% nuFS(mui <400) = 0;
% 
% 
% nuRS = min(nuRS,200);
% nuFS = min(nuFS,200);
% 
% % nuRS = max(nuRS,0);
% % nuFS = max(nuFS,0);
% 
% nus = [nuRS nuFS];

%%


%% no inh correction

% load('ps.mat','pRS')
% nuRS = polyval(pRS,mue');
% nuFS = polyval(pRS,mue');
% nuRS(mue < 500) = 0;
% nuRS(mue > 14000) = 160;
%% inh correction ki = 25

load('psEff.mat','pRSeff')

pRS = pRSeff;
nuRS = polyval(pRS,mue');
nuFS = polyval(pRS,mue');

nuRS(mue < 0) = 0;
nuRS(mue > 14000) = 140;

nuRS = min(nuRS,200);
nuRS = max(nuRS,0);

nus = [nuRS nuFS];

end

