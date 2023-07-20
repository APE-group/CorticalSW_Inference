%% Corresponding Author: Cristiano Capone, cristiano0capone@gmail.com
%  Paper: Simulations Approaching Data: Cortical Slow Waves in Inferred Models of the Whole Hemisphere of Mouse

%% Outer Loop

close all
clear all

for DATASETS = [1]%[1 2 6 9 11 12]%[1 2 3 5 6 7]%
    
    close all
    clearvars -except DATASETS

    
    %%
    nmouse = 1; %DATASETS = [12];
    lambda0 = 1.5;%0.6*1.5;
    
    load(['workSpace_t' num2str(DATASETS) '_mouse' num2str(nmouse) '_phi_pi4_freephi_lam0' num2str(lambda0) '.mat'])
    
    clear W  NuE NuInp
    
    %%
    
    NuE_DATA_All = [];
    
    for nRec = 1:numel(bFactor)
        NuE_DATA_All = [NuE_DATA_All NuE_DATA{nRec}];
    end
    

    %%
    
    timeSingleSet = size(NuE_DATA{nRec},2);
    TIME = timeSingleSet*numel(bFactor);
    
    %%
    
    bfactor_trajectory = [];
    Iextfactor_trajectory = [];
    
    for k = 1:numel(bFactor)
        bfactor_trajectory = [bfactor_trajectory ones(1,timeSingleSet)*bFactor(k)];
        Iextfactor_trajectory = [Iextfactor_trajectory ones(1,timeSingleSet)*IExtFactor(k)];
    end
    
    %%
    
    bfactor_trajectory_smooth = smooth(bfactor_trajectory,700);
    Iextfactor_trajectory_smooth = smooth(Iextfactor_trajectory,700);
    
    figure
    hold on
    plot(bfactor_trajectory,'b')
    plot(bfactor_trajectory_smooth,'b','linewidth',2)
    
    plot(Iextfactor_trajectory,'r')
    plot(Iextfactor_trajectory_smooth,'r','linewidth',2)
    
    xlabel('time step')
    ylabel('factor')
    
    legend('b_fact','b_fact_smooth','I_fact','I_fact_smooth')
    set(gcf,'color','w')
    
    %%
    
    ifPlot = 1; % Se vuoi plottare dinamica e spettro ad ogni simulazione metti 1
    
    k_fact = 1.0;
    b_fact = 1.0;
    
    PERIOD = 1;
    PERIODSlow = 20;
    
    %% Definition of the Exploration Grid
      
    % reply 1 version (28/09/2022)
    D_AMP_vals = linspace(0.,6.,21);
    PERIOD_vals = linspace(1.-0.3158*2 , 7.,22);
    
    
    %% Traiettoria lenta che muove da t*[1] a t*[end]
    
    bFactorInterpolation = linspace(bFactor(1),bFactor(end),TIME);
    IExtFactorInterpolation = linspace(IExtFactor(1),IExtFactor(end),TIME);
    
    %%
    
    if ifPlot ==1
        figure
    end
    
    %%
    
    D_AMPSlow = 0.;
    ecc = 0.;
        
    %%
    
    TIME = 1600;
    
    for K = 1:numel(PERIOD_vals)
        
        K
        
        for J = 11:numel(D_AMP_vals)
            
            addNoise = 2;%2
            D_AMP = D_AMP_vals(J);
            
            PERIOD = PERIOD_vals(K);
            PERIOD_0 = PERIOD;
            
            multNoise = 0;
            AMP0 = 1.;%0.9
            AMP0_b = 1.;
            
            popParam.tauW(1) = tauW;
            
            dt = 0.04;
            
            NuE = abs(randn(Npop,1)*10);
            NuI = zeros(Npop,1);
            W = zeros(Npop,1);
            
            driven = 0;
            phaseExt = 0;
            
            for t = 1:TIME-1+700
                
                if t == TIME/2
                    driven = 0;
                end
                
                if driven
                    NuE_inp(:,t) =  NuE_DATA(:,t);
                else
                    NuE_inp(:,t) =  NuE(:,t);
                end
                
                PERIOD = PERIOD_0*(1 + 0.01*cos(2*pi*dt*t/PERIODSlow) );
                
                random_phase = real(acos(phaseExt(t) + cos(2*pi*dt*t/PERIOD)*2*pi*dt/PERIOD + randn(1)*0.1) );
                
                phaseExt(t+1) = cos(random_phase);
                deformation(t+1) = cos( random_phase*.5);
                
                AMPSlow = (1 + D_AMPSlow*cos(2*pi*dt*t/PERIODSlow) ) ;
                
                AMP = AMP0 + D_AMP*phaseExt(t+1) *(1+ecc*phaseExt(t+1));
                AMP_b = AMP0_b + D_AMP*phaseExt(t+1)*0.5*(1+ecc*phaseExt(t+1));
                
                
                [dersSim,TFESim] = tfFunctionDerParallel0(NuE_inp(:,t),popParam.b(1,:)'.*b_fact.*W(:,t)*AMPSlow*AMP_b,popParam,ExcDrive*AMP , kLat*k_fact);
                NuE(:,t+1) =  randn(Npop,1).*(sqrt(abs(TFESim)/2/popParam.N(1)/dt)*multNoise + addNoise) + TFESim;
                
                I_fact_traj(t) = AMP;
                b_fact_traj(t) = AMP_b;%b_fact*bFactorInterpolation(t)*AMPSlow*AMP_b;
                
                NuE(:,t+1) = max( NuE(:,t+1), 0 );
                
                W(:,t+1) = W(:,t) + dt.*( - W(:,t)./popParam.tauW(1)  + NuE_inp(:,t) );
                
            end
            
            %%
            
            NuE = NuE(:,end-TIME:end);
            
            size(NuE)
            
            %%
            
            %NuE = NuE(:,TIME/2+1:end);
            
            for k = 1:Npop
                NuE(k,:) = smooth(NuE(k,:),1);
            end
            
            %%
            
            Fs = 1/dt;            % Sampling frequency
            % dt = 1/Fs;             % Sampling period
            L = TIME;             % Length of signal
            t = (0:L-1)*dt;        % Time vector
            f = Fs*(0:(L/2))/L;
            
            FFT_SIM = fft(mean(NuE,1));
            FFT_SIM = abs(FFT_SIM/L);
            FFT_SIM = FFT_SIM(1:L/2+1);
            FFT_SIM(2:end-1) = 2*FFT_SIM(2:end-1);
            
            %%
            
            if ifPlot ==1
                
                clf
                
                subplot(2,1,1)
                
                imagesc((1:size(NuE,2))*dt,1:Npop,NuE)
                xlabel('time (s)')
                ylabel('i.d. pop')
                
                title('\nu_e simulation')
                colorbar
                set(gca,'clim',[0 120])
                
                
                subplot(2,1,2)
                hold on
                plot(f,smooth(FFT_SIM))
                %plot(f,smooth(FFT_DATA))
                xlabel('f (Hz)')
                ylabel('psd')
                set(gca,'yscale','log','xscale','log')
                
                colormap(parula)
                drawnow()
                
                set(gcf,'color','w')
                
            end
            
            %%
            
            FFT_SIM_1Hz = smooth(FFT_SIM);
            FFT_SIM_1Hz(find(f<1))=0;
            
            [val,ndx] = max(FFT_SIM_1Hz);
            
            ['SO freq ' num2str(f(ndx),2) 'Hz']
            
            FREQ(K,J) = f(ndx);
            
            %%
            
            mkdir(['ecc0_t'  num2str(DATASETS) '_mouse' num2str(nmouse) '_4' ])
            save(['ecc0_t'  num2str(DATASETS) '_mouse' num2str(nmouse) '_4/NuE_SIM_lognormal_t' num2str(DATASETS) '_DAMP_' num2str(D_AMP,2) '_PERIOD_' num2str(PERIOD,2) '_DeltaAmp_Period_Mouse' num2str(nmouse) '.mat'],'NuE','x_pos_sel','y_pos_sel')
            
        end
     
    end
    
end


