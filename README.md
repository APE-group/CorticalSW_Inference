# CorticalSW_Inference

Code to reproduce results from the [paper](https://arxiv.org/abs/2104.07445):

Capone, Cristiano, et al. "Simulations Approaching Data: Cortical Slow Waves in Inferred Models of the Whole Hemisphere of Mouse." arXiv preprint arXiv:2104.07445 (2021).

Data used are available in the EBRAINS Knowledge Graph [platform](https://kg.ebrains.eu/search/instances/Dataset/28e65cf1-ce13-4c12-92dc-743b0cb66862).

The paper articulates in the following steps:
* Inner Loop: inference parameters though maximum likelihood methd -> to execute this stage run the matlab codes 'InnerLoop.mat' and 'OuterLoop.mat' in the 'Inference' folder
* Data convolution -> (for simulated data) to execute this stage run the code within DataConvolution directory
* Cobrawap pipeline application -> to run thi stage follow instruction within the DataAnalysisWorkflow directory

   The analysis workflow used in this paper is a modified version of [COBRAWAP](https://github.com/INM-6/cobrawap) tool described in this [paper](https://arxiv.org/abs/2211.08527)

* Data post-processing and visualization -> to execute this stage run codes in PostProcessing directory. To do so, analysis results should be organized as follows
  - Output: directory the output of the analysis. It should contain:
    - ConvolvedData: directory containing the convolved data from simulation. Each simulation output should be saved in a directory named "[mouse_number]_t[trial_number]_Ampl_Period"
    - The output of the Cobrawap pipeline on the experimental data saved in directories with name "TI_LENS_[mouse_number]_t[trial_number]"
    - The output of the Cobrawap pipeline on the simulated data grid search saved in directories with name "TI_MF_LENS_[mouse_number]_t[trial_number]_[Almplitude_value]_[Period_value]"
    - The output of the Cobrawap pipeline on the simulated data from inner loop saved in directories with name "TI_MF_LENS_[mouse_number]_t[trial_number]_InnerLoop"

  The output figures will be saved in OutputPlot directory.

