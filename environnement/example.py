if __name__ == "__main__":
    env = CournotEnv(
        n_firms=2,
        a=100,
        b=1,
        costs=[10, 10],
        q_max=100,
        horizon=50,
        seed=42
    )

    state = env.reset()

    for _ in range(5):
        actions = np.array([20.0, 20.0])
        state, rewards, done, info = env.step(actions)
        print(f"Price: {info['price']:.2f}, Rewards: {rewards}")
