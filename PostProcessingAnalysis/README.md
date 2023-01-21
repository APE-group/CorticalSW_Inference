In this directory post-processing analysis is computed.

* ComputeEMD: identifies all the run simulations associated to a single trial for a single mouse (user input) and computes the EMD distance on the histograms of local velocities, directions and inter-spike-interval.
  Inputs: [mouseRec] mouse number, [trial] trial number
  Outputs: computed EMD are saved in the Output/OutputData directory

Codes reproducing figures of the paper are also in this directory.
* Figure2: computes the GMM distributions from a set of experimenta data and the local macroscopic observation histograms, reproducing Figure2 of this [paper](https://arxiv.org/abs/2104.07445https://arxiv.org/abs/2104.07445)
  Inputs: [mouseRec] mouse number, [trial] trial numbers
  Outputs: Figure2 saved in OutputPlot directory
* Figure4: compare experimental and simulated data and the local macroscopic observation histograms, reproducing Figure4 of this [paper](https://arxiv.org/abs/2104.07445https://arxiv.org/abs/2104.07445)
  Inputs: [mouseRec] mouse number, [trial] trial numbers
  Outputs: Figure4 saved in OutputPlot directory
* Figure5: plots rastergrams of travelling wave for experimental data, the output of the inner loop simulation and the output of the outer loop simulation reproducing Figure5 of this [paper](https://arxiv.org/abs/2104.07445https://arxiv.org/abs/2104.07445)
  Inputs: [mouseRec] mouse number, [trial] trial numbers
  Outputs: Figure5 saved in OutputPlot directory
* Figure6: plots mean travelling waves for experimental and simulated data though a GMM analysis, reproducing Figure6 of this [paper](https://arxiv.org/abs/2104.07445https://arxiv.org/abs/2104.07445)
  Inputs: [mouseRec] mouse number, [trial] trial numbers
  Outputs: Figure5 saved in OutputPlot directory
