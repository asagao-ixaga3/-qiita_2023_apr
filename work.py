
import matplotlib.pylab as plt
import numpy as np

from environment import MsdSystem
from normalizer import Normalizer
from uclk import UclkAgent


def run_closed_loop_test_from_fixed_initial_state(agent: UclkAgent, 
    env: MsdSystem, actions, T_simulation, y_initial_raw, normalizer: Normalizer):

    env.reset(y_initial=y_initial_raw)
    while True:
        q = agent.predict_q(normalizer.convert(env.y)) # (na,)
        u = actions[np.argmax(q)] # (,)    
        env.step(u)
        if env.t > T_simulation:
            break

def run_closed_loop_test_from_fixed_initial_state_without_force(agent: UclkAgent, 
    env: MsdSystem, actions, T_simulation, y_initial_raw, normalizer: Normalizer):

    env.reset(y_initial=y_initial_raw)
    while True:
        q = agent.predict_q(normalizer.convert(env.y)) # (na,)
        u = 0.0
        env.step(u)
        if env.t > T_simulation:
            break

def create_prediction_chart(S0,S1,S2,fig):
    # S0,S1,S2: (*,2)

    ax1,ax2 = fig.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    UV10 = S1-S0 # (n,2)
    UV20 = S2-S0

    for (ax, S0, S1) in [(ax1,S0,S1), (ax2,S0,S2)]:
        ax.plot(*S0.T, "ks", label = "initial state", markerfacecolor="none", 
                markersize = 12)
        ax.plot(*S1.T, "ks", label = "next state", markerfacecolor="k", 
                markersize = 12)
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)

    ax1.quiver(*S0.T, *UV10.T, color = "gray", label = "transition", scale=1, 
                angles = "xy", scale_units = "xy")
    ax2.quiver(*S0.T, *UV20.T, color = "gray", label = "transition", scale=1, 
                angles = "xy", scale_units = "xy")

    ax1.set_title("simulator's transition")
    ax2.set_title("trained agent's transition")

    for ax in (ax1,ax2):
        ax.legend()
        ax.grid()
        ax.set_xlabel("position")
        ax.set_ylabel("velocity")

    fig.tight_layout()

def create_value_distribution_chart(V0,V1,V2,labels,fig):
    # V0,V1,V2: (#groups)
    # labels: (#groups)

    n_groups = len(V0)

    # create plot
    (ax1,ax2) = fig.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    index = np.arange(n_groups)
    bar_width = 0.35

    for (ax,X,Y,title) in [(ax1,V0,V1,"Simulator's transition"), 
                    (ax2,V0,V2, "Trained agent's transition")]:
        ax.bar(index, X, bar_width,
            color='white',
            edgecolor = "black",
            label='at initial state')

        ax.bar(index + bar_width, Y, bar_width,
            color='k',
            label='at next state')

        ax.set_xlabel('')
        ax.set_ylabel('Value')
        ax.set_title(title)

        ax.set_xticks(index + bar_width/2, labels)
        ax.legend()
        ax.grid()

    fig.tight_layout()

class Work(object):

    na = 2
    actions = np.linspace(-3, 3, na) # (na,)
    n_seed = None
    n_step = 3
    N_train = 2**7

    env = MsdSystem()
    normalizer = Normalizer(env)
    T_simulation = 5.0
    n_seed = 3
    weight_mv = 0.0

    def reward(self, Y):
        # S: (...,nx)
        Y_raw = self.normalizer.invert(Y)
        Position = Y_raw[...,0] # (...)
        Error = np.abs(Position) # (...)
        cost = Error[:,None] + self.weight_mv * np.abs(self.actions) # (...,na)
        Reward = -cost # (...,na)
        return Reward

    def collect_data(self, N):

        nx = self.env.nx
        na = self.na

        S0 = np.zeros((N, nx)) # (N, nx)
        A0 = np.random.randint(low=0, high=na, size=(N,)) # (N,)
        S1 = np.zeros((N, nx)) # (N, nx)
        for i in range(N):
            self.env.reset()
            S0[i,:] = self.normalizer.convert(self.env.y)
            self.env.step(self.actions[A0[i]])            
            S1[i,:] = self.normalizer.convert(self.env.y) # (nx,)

        return S0, A0, S1
    
    def train(self, feat_map_params, training_params, S0, A0, S1):

        nx = self.env.nx
        na = self.na

        agent = UclkAgent(nx=nx,na=na,**feat_map_params)
        
        S_tld = S0.copy() # (N, nx)
        R_tld = self.reward(S_tld) # (N, na)

        agent.fit(S0, A0, S1, S_tld, R_tld, **training_params)

        return agent

    def run(self, debug_mode = True):

        if debug_mode:
            N = 2**1
        else:
            N = self.N_train

        n_rounds = 2**0
        gamma = 0.9
        n_itr = 2**4
        p_dim = 2**7
        v_dim = 2**7
        p_scale = 0.4
        v_scale = 0.4

        S0, A0, S1 = self.collect_data(N)

        training_params = dict(n_rounds=n_rounds, gamma = gamma, 
            n_itr = n_itr)
        feat_map_params = dict(p_scale = p_scale, p_dim = p_dim,
                                v_dim = v_dim, v_scale = v_scale)
        agent = self.train(feat_map_params, training_params, S0, A0, S1)
        self.evaluate(agent, S0, A0)
                
    def evaluate(self, agent: UclkAgent, S0, A0):

        fig = plt.figure(figsize=[12, 8])
        fig_name = "./img/closed_loop_test.png"      

        n_fig = 4
        for i, y_initial_raw in enumerate([(1,0), (-1,0), (0,1), (0,-1)]):
            ax = fig.add_subplot(n_fig, 1, i+1)

            run_closed_loop_test_from_fixed_initial_state(agent, self.env, 
                self.actions, self.T_simulation, y_initial_raw=y_initial_raw
                , normalizer=self.normalizer)
            P = np.stack(self.env.log["Position"], axis=0) # (*)
            V = np.stack(self.env.log["Velocity"], axis=0) # (*)
            F = np.stack(self.env.log["Force"], axis=0) # (*)
            T = np.stack(self.env.log["Time"], axis=0) # (*)

            ax.set_title("Initial state Pos = %.2f Vel = %.2f" % (P[0], V[0]))
            ax.plot(T, P, label = "Position")
            ax.plot(T, V, label = "Velocity")
            ax.plot(T, F, label = "Force")
            ax.set_ylim(-3,3)
            ax.legend()
            ax.grid()
        
        fig.tight_layout()
        fig.savefig(fig_name)
        plt.close(fig)


        fig = plt.figure(figsize=[12, 8])
        fig_name = "./img/open_loop_simulation.png"      

        n_fig = 4
        for i, y_initial_raw in enumerate([(1,0), (-1,0), (0,1), (0,-1)]):
            ax = fig.add_subplot(n_fig, 1, i+1)

            run_closed_loop_test_from_fixed_initial_state_without_force(agent, 
                self.env, self.actions, self.T_simulation, 
                y_initial_raw=y_initial_raw, normalizer=self.normalizer)
            P = np.stack(self.env.log["Position"], axis=0) # (*)
            V = np.stack(self.env.log["Velocity"], axis=0) # (*)
            F = np.stack(self.env.log["Force"], axis=0) # (*)
            T = np.stack(self.env.log["Time"], axis=0) # (*)

            ax.set_title("Initial state Pos = %.2f Vel = %.2f" % (P[0], V[0]))
            ax.plot(T, P, label = "Position")
            ax.plot(T, V, label = "Velocity")
            ax.plot(T, F, label = "Force")
            ax.set_ylim(-3,3)
            ax.legend()
            ax.grid()
        
        fig.tight_layout()
        fig.savefig(fig_name)
        plt.close(fig)

        n_grid = 16

        x = np.linspace(-2,2,n_grid)
        v = np.linspace(-2,2,n_grid)

        X, V = np.meshgrid(x,v) # (ng,ng)
        XV = np.stack((X,V),axis=-1).reshape(-1,2) # (ng*ng,2)
        Y_raw = self.normalizer.invert(np.stack((X,V), axis=-1)) # (ng,ng,2)
        Qfunc = agent.predict_q(XV).reshape(n_grid,n_grid,-1) # (ng,ng,na)
        Vfunc = agent.predict_v(XV).reshape(n_grid,n_grid) # (ng,ng)

        fig = plt.figure(figsize=[8,6])
        fig.clf()

        ax = fig.add_subplot(1,1,1)
        im = ax.pcolor(Y_raw[:,:,0],Y_raw[:,:,1],Vfunc)
        plt.colorbar(im)
        ax.axhline(y=0, linestyle="--", color="white")
        ax.axvline(x=0, linestyle="--", color="white")
        ax.set_title("State value function heat map")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")

        fig.tight_layout()

        fig.savefig("./img/state_value_function.png")

        fig = plt.figure(figsize=[8,6])
        fig.clf()

        ax = fig.add_subplot(1,1,1)
        im = ax.pcolor(Y_raw[:,:,0],Y_raw[:,:,1],Qfunc[:,:,1]-Qfunc[:,:,0], 
                       cmap = "bwr")
        plt.colorbar(im)
        ax.axhline(y=0, linestyle="--", color="white")
        ax.axvline(x=0, linestyle="--", color="white")
        ax.set_title("q(s,a=1)-q(s,a=0)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")

        fig.tight_layout()
        fig.savefig("./img/action_value_function.png")

        S0_raw = np.array([
            (0,0),
            (1,0),
            (-1,0),
            (0,1),
            (0,-1)]) # (5,2)
        A0 = np.ones(5, dtype=np.int64) # (5,)
        S0 = self.normalizer.convert(S0_raw) # (*,2)
        S1_raw = np.zeros(S0_raw.shape)
        S2_raw = self.normalizer.invert(agent.predict_next_state(S0, A0))
        for i in range(S0_raw.shape[0]):
            self.env.reset(y_initial=S0_raw[i,:])
            self.env.step(self.actions[A0[i]])
            S1_raw[i,:] = self.env.y

        V0 = agent.predict_v(self.normalizer.convert(S0_raw)) # (*,)
        V1 = agent.predict_v(self.normalizer.convert(S1_raw)) # (*,)
        V2 = agent.predict_v_at_next_step(S0,A0) # (*,)

        fig = plt.figure(figsize=[10,6])        
        create_prediction_chart(S0_raw,S1_raw,S2_raw,fig)
        fig.tight_layout()
        fig.savefig("./img/flow.png")
        plt.close(fig)

        fig = plt.figure(figsize=[10,6])
        labels = ('(0,0)', '(1,0)', '(-1,0)', '(0,1)', '(0,-1)')
        create_value_distribution_chart(V0,V1,V2,labels,fig)
        fig.savefig("./img/value_distributions.png")
        plt.close(fig)