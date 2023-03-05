from Helper import LearningCurvePlot

def write_rewards(exp_name, rewards: list) -> None:
    rewards = ["#t,r"] + [f"{t},{r}\n" for t, r in enumerate(rewards)]
    with open(f"timestep_rewards_{exp_name}", "w") as f:
        f.writelines(rewards)

def plot_rewards_csv(fn: str) -> None:
    rewards = []
    with open(fn, "r") as f:
        rewards = f.readlines()

    if not rewards:
        return
    
    # cut off header
    rewards = rewards[1:]
    