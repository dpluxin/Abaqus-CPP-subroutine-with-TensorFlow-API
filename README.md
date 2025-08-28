To use GCC compiler for Abaqus CPP user subroutine, the following tensorflow package (built from source) is used:
"Version	        Python version	Compiler	  Build tools
tensorflow-2.12.0	3.8-3.11	      GCC 9.3.1	  Bazel 5.3.0	8.6	11.8"
(https://www.tensorflow.org/install/source)

You can also download the pre-built object files and required packages from:

https://drive.google.com/file/d/1SI53XGZTNCkP4Y2z2TsK3p4c5ZM2ybZj/view?usp=drive_link

In Linux OS, copy '/include' to '/usr/local/include' and "libtensorflow_cc.so.2" and "libtensorflow_framework.so.2" to '/usr/local/lib'.
Create Symbolic Links for Shared Libraries:
sudo ln -s libtensorflow_cc.so.2 libtensorflow_cc.so
sudo ln -s libtensorflow_framework.so.2 libtensorflow_framework.so

The abaqus_v6.env defines the include directory and linking against the shared objects.
