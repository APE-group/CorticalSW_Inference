function out = PhiLin(mu, sigma2, beta, h, theta, tarp)
%
% out = PhiLin(mu, sigma2, beta, h, theta, tarp)
%
   mu = mu - beta;
   csi = mu./sigma2;
   out = 1./(1./(2*csi.*mu).*(exp(-2*theta.*csi)-exp(-2*h.*csi)) + (theta-h)./mu + tarp);

