function NuOut = EffectivePhi(NuIn, Ca, Net)
%
% NuOut = EffectivePhi(NuIn, Ca, Net)
%

FIXED_POINT_TOLERANCE = 5e-6;
DAMPENING_FACTOR = 0.1;
% MAX_ITERATIONS = 1000;
MAX_ITERATIONS = 20000;

if numel(Net.ndxEFg) > 1
   error('[EffectivePhi] More than 1 foreground population');
end

GC = Net.SNParam.GC(Net.ndxEFg);
Net.SNParam.GC(Net.ndxEFg) = 0;

NuOut = zeros(Net.P,numel(NuIn));
for nt = numel(NuIn):-1:1
   if nt == numel(NuIn)
      Nu0 = Net.SNParam.Nu;
   end
   Nu0(Net.ndxEFg) = NuIn(nt);
   Nu = Nu0;
   
   nr = 0;
   while nr < MAX_ITERATIONS
      lMu = Mu(Nu0, Net);
      lSigma2 = Sigma2(Nu0, Net);
      
      for np = 1:Net.P
         if np ~= Net.ndxEFg
            Nu(np) = feval(Net.SNParam.Phi{np}, lMu(np), lSigma2(np), Net.SNParam.Beta(np), Net.SNParam.H(np), Net.SNParam.Theta(np), Net.SNParam.Tarp(np));
         end
      end
      
      if norm(Nu-Nu0) < FIXED_POINT_TOLERANCE
         break
      else
         Nu0 = DAMPENING_FACTOR * (Nu - Nu0) + Nu0;
      end
      
      nr = nr + 1;
   end % while nr < MAX_ITERATIONS
   
   if nr == MAX_ITERATIONS
      warning(['[EffectivePhi] Effective function not accurately evaluated (NuIn=' num2str(NuIn(nt)) ')'])
   end
   
   lMu = Mu(Nu, Net);
   lMu( Net.ndxEFg) = lMu(Net.ndxEFg) - GC * Ca;
   lSigma2 = Sigma2(Nu, Net);
   for np = 1:Net.P
      NuOut(np,nt) = feval(Net.SNParam.Phi{np}, lMu(np), lSigma2(np), Net.SNParam.Beta(np), Net.SNParam.H(np), Net.SNParam.Theta(np), Net.SNParam.Tarp(np));
   end
end % for nt = ...

Net.SNParam.GC(Net.ndxEFg) = GC;
