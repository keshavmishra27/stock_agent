#import libraries
import pandas as pd
import numpy as np
#read the data
data=pd.read_csv(r"YNDX_150101_151231.csv")
print(data.head())#each row repesents one candle
#rename the collumns
collums={"<DATE>":"DATE",
         "<TIME>":"TIME",
         "<OPEN>":"OPEN",
         "<HIGH>":"HIGH",
         "<CLOSE>":"CLOSE",
         "<LOW>":"LOW",
         "<VOL>":"VOL"
}
data_new=data.rename(columns=collums)

#check the changed data
print(data_new.head())

#delete unnessaccary collumns
del data_new["DATE"],data_new["TIME"],data_new["VOL"]
print(data_new.head())
print(data_new.info())
#removing unnecessary very small data
data_new['remove']=data_new.apply(lambda row:all([abs(i-row[0])<1e-8 for i in row]),axis=1)#explain this line
data2=data_new.query("remove==False").reset_index(drop=True)
del data_new["remove"]
print(data2.describe())
print(data2.info())

#normalize the data
data2["HIGH"]=(data2['HIGH']-data2["OPEN"])/data2["OPEN"]
data2["LOW"]=(data2['LOW']-data2["OPEN"])/data2["OPEN"]
data2["CLOSE"]=(data2['CLOSE']-data2["OPEN"])/data2["OPEN"]

print(data2.head())

#create YAND env
class Yand():
    def __init__(self,data,obs_bars=10,test=False,commission_perc=0.3):
        self.data=data
        self.obs_bars=obs_bars
        self.have_pos=0#tells if we put a position previously and we closed yet or not
        self.test=test
        self.commission_perc=commission_perc

        if test==False:#twill start episode from random step, i.e start from random candle in dataset
            self.curr_step=np.random.choice(self.data.HIGH.shape[0]-self.obs_bars*10)+self.obs_bars# self.data.high.shape[0] represents no. of rows or no. of candles

        
        else:#testing
            self.curr_step=self.obs_bars#will start from the begining of the datset

        self.state=self.data[self.curr_step-self.obs_bars : self.curr_step]

    def step(self,action):
        reward=0
        done=False
        relative_close=self.state["CLOSE"][self.curr_step-1]#estracting the close price for the last candle stick in our state
        open=self.state["OPEN"][self.curr_step-1]
        close=open*(1+relative_close)


        if action=="buy" and self.have_pos==False:
            self.have_pos=True
            self.open_price=close
            reward=self.commission_perc

        elif action=="sell" and self.have_pos==True:
            reward=self.commission_perc
            if self.test==False:
                done=True

            reward+=100.0*(close-self.open_price)/self.open_price
            self.have_pos=False
            self.open_price=0.0

        self.curr_step+=1
        self.state=self.data[self.curr_step-self.obs_bars: self.curr_step]

        if self.curr_step==len(self.data):
            done=True

        #preparing state for for Qnetwork
        state=np.zeros((5,self.obs_bars),dtype=np.float32) #2d array, length=5, height=obs_bars,type=float32
        state[0]=self.state.HIGH.to_list()#
        state[1]=self.state.LOW.to_list()
        state[2]=self.state.CLOSE.to_list()
        state[3]=int(self.have_pos)

        if self.have_pos:
            state[4]=(close-self.open_price)/self.open_price
        
        return state,reward,done

actions={0:"do_nothing",1:"buy",2:"close"}

Yand_env=Yand(data2,test=False,obs_bars=30)
state,reward,done=Yand_env.step("do nothing")
print(state.shape)
input_shape=state.shape
    

        

        


