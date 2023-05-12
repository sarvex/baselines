import gym

from baselines import deepq


def callback(lcl, _glb):
    return lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) >= 19900


def main():
    env = gym.make("CartPole-v0")
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
