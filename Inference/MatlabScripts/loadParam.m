function Net = loadParam(InSingleNeuron,InCoupling)

sn = load(InSingleNeuron);
c = load(InCoupling);

Net.SNParam.N = sn(:,1);
Net.SNParam.Nu = sn(:,2);
Net.SNParam.Beta = sn(:,3);
Net.SNParam.Theta = sn(:,4);
Net.SNParam.H = sn(:,5);
Net.SNParam.Tarp = sn(:,6);
Net.SNParam.NExt = sn(:,7);
Net.SNParam.NuExt = sn(:,8);
Net.SNParam.JExt = sn(:,9);
Net.SNParam.DeltaExt = sn(:,10);
Net.SNParam.Type = sn(:,11);
Net.SNParam.DMin = sn(:,12);
Net.SNParam.TauD = sn(:,13);

Net.P = size(sn,1);

for n = 1:Net.P
   if Net.SNParam.Type(n) == 0
      Net.SNParam.Phi{n} = @PhiLin;
   else
      if Net.SNParam.Type(n) == 1
         Net.SNParam.Phi{n} = @PhiExpFromLUT;
      else
         if Net.SNParam.Type(n) == 2
            Net.SNParam.Phi{n} = @PhiExp;
         else
            disp(['ERROR: Unknown neuron type (line ' num2str(n) ',col. 11).']);
            return;
         end
      end
   end
end

Net.CParam.c = c(1:Net.P,:);
Net.CParam.J = c((Net.P+1):(2*Net.P),:);
Net.CParam.Delta = c((2*Net.P+1):(3*Net.P),:);
