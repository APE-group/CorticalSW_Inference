function out = PhiExpFromLUT(mu, sigma2, tau, h, theta, tarp)
%
% out = PhiExpFromLUT(mu, sigma2, tau, h, theta, tarp)
%

%mfpt = tau*sqrt(pi)*quad('exp(x.^2).*(1 + erf(x))', (h-mu)/sigma, (theta-mu)/sigma);

global PhiExpLUT

if isempty(PhiExpLUT)
   load('PhiExpLUT.mat');
end

sigma = sqrt(sigma2);
for n = 1:length(mu)
 
   a = (h(n) - mu(n) * tau(n)) / (sigma(n) * sqrt(tau(n)));
   b = (theta(n) - mu(n) * tau(n)) / (sigma(n) * sqrt(tau(n)));
   logba = log(b-a);
   
   if logba > PhiExpLUT.logbaMax | logba < PhiExpLUT.logbaMin
      disp(sprintf('Warning (PhiExpFromLUT): log(b-a)=%g out of LUT range.', log(b-a)));
   end

   if b > PhiExpLUT.bMax 
      out(n) = 0;
   else
      if b < PhiExpLUT.bLowHigh
         if b < PhiExpLUT.bMin
            disp(sprintf('Warning (PhiExpFromLUT): b=%g too lower for LUT.', b));
            out(n) = 1 / tarp(n);
         else
            out(n) = 1 / (tau(n) * sqrt(pi) * ...
                     exp(fnval(PhiExpLUT.cslow, {b, logba})) + tarp(n));
         end
      else
         out(n) = 1 / (tau(n) * sqrt(pi) * ...
                  exp(fnval(PhiExpLUT.cshigh, {b, logba})) + tarp(n));
      end
   end
end
