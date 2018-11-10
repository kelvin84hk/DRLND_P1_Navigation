import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1=nn.Conv2d(state_size,6,kernel_size=8,stride=2) # 37 x 37
        self.conv2=nn.Conv2d(6,12,kernel_size=5,stride=2) # 17 x 17 
        self.conv3=nn.Conv2d(12,24,kernel_size=3,stride=2) # 8 x 8

        ###############################
        self.dense4=nn.Linear(int(8*8*24),int(8*8*24/2))
        self.dense5=nn.Linear(int(8*8*24/2),action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x=F.relu(self.conv1(state))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.reshape(x.size(0),-1)

        x=F.relu(self.dense4(x))
        x=self.dense5(x)
        return x
    
class QNetworkDuel(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkDuel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size=action_size
        ##############################
        self.conv1=nn.Conv2d(state_size,2,kernel_size=4,stride=2, padding=1)
        self.conv2=nn.Conv2d(2,4,kernel_size=4,stride=2, padding=1)
        self.conv3=nn.Conv2d(4,8,kernel_size=4,stride=2, padding=1)
        ###############################
        self.dense14=nn.Linear(int(10*10*8),int(10*10*8/2))
        self.dense15=nn.Linear(int(10*10*8/2),1)
        ###############################
        self.dense24=nn.Linear(int(10*10*8),int(10*10*8/2))
        self.dense25=nn.Linear(int(10*10*8/2),action_size)

    def forward(self, state):
        
        x=F.relu(self.conv1(state))
        #x=self.pool1(x)
        x=F.relu(self.conv2(x))
        #x=self.pool2(x)
        x=F.relu(self.conv3(x))
        #x=self.pool3(x)
        x=x.reshape(x.size(0),-1)

        x1=F.relu(self.dense14(x))
       # x1=F.relu(self.dense15(x1))
       # x1=F.relu(self.dense16(x1))
        v=self.dense15(x1).expand(x1.size(0),self.action_size)
        
        x2=F.relu(self.dense24(x))
        #x2=F.relu(self.dense25(x2))
        #x2=F.relu(self.dense26(x2))
        adv=self.dense25(x2)
        x3=v+adv-adv.mean(1).unsqueeze(1).expand(x2.size(0),self.action_size)
        
        return x3
        
class PolicyNet(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(PolicyNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size=action_size
        ##############################
        self.conv1=nn.Conv2d(state_size,6,kernel_size=8,stride=2) # 37 x 37
        self.conv2=nn.Conv2d(6,12,kernel_size=5,stride=2) # 17 x 17 
        self.conv3=nn.Conv2d(12,24,kernel_size=3,stride=2) # 8 x 8 
        ###############################
        self.dense4=nn.Linear(int(8*8*24),int(8*8*24/2))
        self.dense5=nn.Linear(int(8*8*24/2),int(8*8*24/4))
        self.dense6=nn.Linear(int(8*8*24/4),action_size)

    def forward(self, state):
        
        x=F.relu(self.conv1(state))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.reshape(x.size(0),-1)

        x=F.relu(self.dense4(x))
        x=F.relu(self.dense5(x))
        x=F.softmax(self.dense6(x),1)
                
        return x


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed,fc1_units=256,fc2_units=128): 
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """    
        super(Actor, self).__init__()  
        self.seed = torch.manual_seed(seed)    
        self.conv1=nn.Conv2d(state_size,6,kernel_size=8,stride=2) # 37 x 37
        self.b1=nn.BatchNorm2d(6)
        self.conv2=nn.Conv2d(6,12,kernel_size=5,stride=2) # 17 x 17
        self.b2=nn.BatchNorm2d(12) 
        self.conv3=nn.Conv2d(12,24,kernel_size=3,stride=2) # 8 x 8 
        self.b3=nn.BatchNorm2d(24)
        self.dense4=nn.Linear(int(8*8*24),fc1_units)
        #self.b4=nn.BatchNorm1d(fc_units)
        #self.dense5=nn.Linear(fc1_units,fc2_units)
        self.dense6=nn.Linear(fc1_units,action_size)
    
    def forward(self, state):
        
        x=F.relu(self.conv1(state))
        x=self.b1(x)
        x=F.relu(self.conv2(x))
        x=self.b2(x)
        x=F.relu(self.conv3(x))
        x=self.b3(x)
        x=x.reshape(x.size(0),-1)
        x=F.relu(self.dense4(x))
        #x=self.b4(x)
        #x=F.relu(self.dense5(x))
        x=F.softmax(self.dense6(x),1)
                
        return x        

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed,fc1_units=128,fc2_units=128): 
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimesion of each action
            seed (int): Random seed
        """    
        super(Critic, self).__init__()  
        self.seed = torch.manual_seed(seed)    
        self.conv1=nn.Conv2d(state_size,6,kernel_size=8,stride=2) # 37 x 37
        #self.b1=nn.BatchNorm2d(6)
        self.conv2=nn.Conv2d(6,12,kernel_size=5,stride=2) # 17 x 17 
        #self.b2=nn.BatchNorm2d(12)
        self.conv3=nn.Conv2d(12,24,kernel_size=3,stride=2) # 8 x 8 
        #self.b3=nn.BatchNorm2d(24)
        self.dense4=nn.Linear(int(8*8*24)+action_size,fc1_units)
        #self.b4=nn.BatchNorm1d(fc1_units)
        self.dense5=nn.Linear(fc1_units,fc2_units)
        self.dense6=nn.Linear(fc2_units,1)
    
    def forward(self, state,action):
        
        xs=F.relu(self.conv1(state))
        #xs=self.b1(xs)
        xs=F.relu(self.conv2(xs))
        #xs=self.b2(xs)
        xs=F.relu(self.conv3(xs))
        #xs=self.b3(xs)
        xs=xs.reshape(xs.size(0),-1)
        x = torch.cat((xs, action), dim=1)
        x=F.relu(self.dense4(x))
        #x=self.b4(x)
        x=F.relu(self.dense5(x))
        x=self.dense6(x)

        return x                

class CriticD4PG(nn.Module):
    """Critic (distribution) Model."""

    def __init__(self, state_size, action_size, seed,fc1_units=128,fc2_units=128, n_atoms=51, v_min=-1, v_max=1): 
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimesion of each action
            seed (int): Random seed
        """    
        super(CriticD4PG, self).__init__()  
        self.seed = torch.manual_seed(seed)    
        self.conv1=nn.Conv2d(state_size,6,kernel_size=8,stride=2) # 37 x 37
        self.b1=nn.BatchNorm2d(6)
        self.conv2=nn.Conv2d(6,12,kernel_size=5,stride=2) # 17 x 17 
        self.b2=nn.BatchNorm2d(12)
        self.conv3=nn.Conv2d(12,24,kernel_size=3,stride=2) # 8 x 8 
        self.b3=nn.BatchNorm2d(24)
        self.dense4=nn.Linear(int(8*8*24)+action_size,fc1_units)
        #self.b4=nn.BatchNorm1d(fc1_units)
        self.dense5=nn.Linear(fc1_units,fc2_units)
        self.dense6=nn.Linear(fc2_units,n_atoms)
        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

    def forward(self, state,action):
        
        xs=F.relu(self.conv1(state))
        xs=self.b1(xs)
        xs=F.relu(self.conv2(xs))
        xs=self.b2(xs)
        xs=F.relu(self.conv3(xs))
        xs=self.b3(xs)
        xs=xs.reshape(xs.size(0),-1)
        x = torch.cat((xs, action), dim=1)
        x=F.relu(self.dense4(x))
        #x=self.b4(x)
        x=F.relu(self.dense5(x))
        x=self.dense6(x)

        return x

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)                        