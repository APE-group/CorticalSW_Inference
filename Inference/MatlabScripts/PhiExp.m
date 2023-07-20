function out = PhiExp(mu, sigma2, tau, h, theta, tarp)

%mfpt = tau*sqrt(pi)*quad('exp(x.^2).*(1 + erf(x))', (h-mu)/sigma, (theta-mu)/sigma);

sigma = sqrt(sigma2);
for n = 1:length(mu)
   a = (h(n) - mu(n) * tau(n)) / (sigma(n) * sqrt(tau(n)));
   b = (theta(n) - mu(n) * tau(n)) / (sigma(n) * sqrt(tau(n)));
   if b > 5 
      out(n) = 0;
   else
      out(n) = 1 / (tau(n) * sqrt(pi) * quad(@auxPhiExp, a, b) + tarp(n));
   end
end

