Data convolution

Data are convoluted with a log-normal function to extract the calcium dynanimic from spiking activity.
In the [paper](https://www.overleaf.com/project/637f7929689c8fb0decb6a0b), this is used to convolve the simulated dataset fter the inner loop procedure.

To run this section simply execute:

```python
python Convolve.py --load_dir <path_to_data> --save_dir <name_of_output_directory>
```
data will be saved in the directory Output/ConvolvedData


