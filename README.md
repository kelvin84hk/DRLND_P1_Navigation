[image1]:https://github.com/kelvin84hk/DRLND_P1_Navigation/blob/master/pics/dq.gif "Trained Agent"


# Project 1 : Navigation 

## Project Details:

For this project, an agent is trained to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +15 over 100 consecutive episodes.

## Getting Started:

To run the code, you need to have PC Windows (64-bit) with Anaconda with Python 3.6 installed.

To download Anaconda, please click the link below:

https://www.anaconda.com/download/

Clone or download and unzip the P1_Navigation folder.

Download by clicking the link below and unzip the environment file under P1_Navigation folder

https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip

Download by clicking the link below and unzip the ml-agents file under P1_Navigation folder

https://github.com/Unity-Technologies/ml-agents/tree/0.4.0b

### Dependencies :

To set up your Python environment to run the code in this repository, follow the instructions below.

  1. Create (and activate) a new environment with Python 3.6.
  
 ```
 conda create --name drlnd python=3.6
 activate drlnd
 ```
  2. Install Pytorch by following the instructions in the link below.
  
     https://pytorch.org/get-started/locally/
    
  3. Then navigate to P1_Navigation/ml-agents-0.4.0b/python and install ml-agent.
     ```
     pip install .
     ```
  4. Install matplotlib for plotting graphs.
     ```
     conda install -c conda-forge matplotlib
     ```
  5. (Optional) Install latest prompt_toolkit may help in case Jupyter Kernel keeps dying
     ```
     conda install -c anaconda prompt_toolkit 
     ```
     
## Run the code 

  Open Navigation.ipynb in Jupyter and press Ctrl+Enter to run the first cell to import all the libraries.
  
  ### Watch a random agent
   Run 2nd cell  to 6th cell to watch how a random agent plays.
   
  ### Train an agent
   Run the cells which contain the "train" function and then choose those cells with "train an agent with ..." in comment
   
  ### Watch a trained agent
   Run the cells with "watch a ... trained agent" to watch how an agent trained by a particular algorithm behaves.
