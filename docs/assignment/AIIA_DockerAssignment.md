<!-- page 1/2 -->

Deployment 
of 
Deep 
Learning 
Architectures in Docker Containers 
 
Objective 
The objective of this assignment is for students to gain experience in selecting, configuring 
and deploying Deep Learning architectures within Docker containers. 
 
Deliverable 
A short report (maximum 6 pages excluding cover and table of contents) describing the 
work done including the code and commands used to create and run the containers should 
be delivered. Just include a brief description of the selected model you work with. 
 
Docker Installation 
Docker should be installed according to the instructions on the official website or use cloud 
services to deploy the images. Docker is compatible with Linux, MacOS and Windows, 
however as explained in the theory slides it is recommended to use Linux. Also, if you want 
to use NVIDIA Container Toolkit to be able to make use of GPUs it will be a prerequisite. 
 
Description of the assignment 
In this assignment a Docker container capable of running a Deep Learning architecture to 
perform inference on a pre-trained model must be assembled. The choice of architecture 
and ML/DL framework is free. Once the architecture has been assembled, at least one 
inference must be done in a pre-trained model.  
 
A graphic interface must be used for doing the inference. Additionally, the interface must be 
isolated from the inference using different containers with Docker Compose. Optionally you 
can train a model for a few epochs and/or publish the model in Hugging Face Spaces so that 
inference can be done through an API. 
 
Assignment Considerations 
●​ Many architectures with available code have public Dockerfiles available for 
configuration. 
●​ Improvements/modifications/self-definition of Dockerfiles in architectures where it is 
not available will be positively considered. 
●​ The order of definition of the Dockerfile should be taken into account to make proper 
use of the cache in Docker and optimize the space used by the images.

<!-- page 2/2 -->

●​ Sometimes the installation of some packages is designed for an interactive 
environment (a classic linux terminal), when defining the instructions in the Dockerfile 
it is important to take into account flags of the type “-y” to automatically accept the 
installations, and/or add the argument DEBIAN_FRONTEND=noninteractive at the 
beginning of the Dockerfile so that it does not ask for user interaction during the 
installation. 
●​ For the execution of the container, the use of parameters that control will be valued: 
○​ The graphics card used 
○​ The RAM limit 
○​ The use of shared memory 
○​ Etc. 
In addition to the container being ready for training or deployment in Hugging Face 
Spaces as mentioned above. 
●​ It is mandatory that there is a well-defined logical separation for the various 
containers. One will only be used for inference and the other will only serve as a 
graphical interface, you may use more depending on your application if necessary. To 
separate the interface from the model inference using Docker Compose, it is 
recommended to use at least two containers: 
○​ The previously made inference container with a model. To be able to make 
requests to this container it will be necessary to encapsulate the inference 
function in a web server (flask, FastAPI).   
○​ A container in which only a very simple graphical interface (we recommend 
using gradio) runs and makes requests to the other container. 
 
Evaluation 
We will consider the following aspects when grading your assignments: 
●​ Quality of the report: structure, number of pages (do not exceed), etc. 
●​ Performing inference (locally and through GUI) within a Docker container on any 
model (results from the terminal are fine). 
●​ Quality of the Dockerfile: instructions used, leveraging existing images, cache layers, 
etc. 
●​ Detail of the parameters used to run the container. 
●​ Docker Compose structure to perform graphical inference using a local API as 
described before. 
●​ To get the maximum grade at least some of the following requirements should be 
met: 
○​ NVIDIA container toolkit + training and inference on GPU 
○​ Upload to Hugging Face spaces and use of online API inference. 
○​ Any other improvements to the Dockerfile, Docker Compose YAML file or 
parameters to run the container besides the explained ones. 
 
Submission 
The report must be submitted as a PDF file through Moodle.
