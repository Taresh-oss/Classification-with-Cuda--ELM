# Classification-with-Cuda--ELM
This repository contains two python notebooks one is for CIFAR 10 dataset and the other one being CIFAR 100 dataset.

The main idea of this project is working with CUDA to speed up the classification of the provided dataset while using ELM as the classifying algorithm.
No specific library has been used to create ELM. The twist here is that the code of ELM is written using Cuda. The major operations that had to be looked at 
inorder to create CUDA BASED ELM is matrix multiplication, matrix addition and Moore pseudo Inverse of a matrix which in itself is a tideous task.

Steps I followed to setup cuda in Google Colab.

Some packages have to be installed in the order provided below.

1. Change the runtime to GPU by clicking on Runtime and then chnage runtime type.
2. uninstall all the older versions of cuda using:
	!apt-get --purge remove cuda nvidia* libnvidia-*
	!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
	!apt-get remove cuda-*
	!apt autoremove
	!apt-get update

3. Installing Cuda Version 9, using commmand:
	!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
	!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
	!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
	!apt-get update
	!apt-get install cuda-9.2

4.Check the Version of CUDA by:
	!nvcc --version	

5. Run command to set nvcc in notebook:
	!pip install git+git://github.com/andreinechaev/nvcc4jupyter.git

6. Load the extension using :
	%load_ext nvcc_plugin

7. Now book.h file has to be uploaded to the sample_data folder in the drive.

Now, Uploading the file to the notebook.

8. The code is good to run.
