import slimevolleygym

# Create the environment
env = slimevolleygym.make("SlimeVolley-v0")

# Reset the environment to get the initial observation
observation = env.reset()

# Example: Run a few random steps
for _ in range(100):
    action = env.action_space.sample()  # Random action
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()

env.close()


