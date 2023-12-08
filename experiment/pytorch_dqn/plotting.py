import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import torch


def plot_durations(episode_durations: list[int], show_result: bool = False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_scores(episode_scores: list[dict[int, float]], show_result: bool = False):
    if len(episode_scores) == 0:
        raise ValueError("The list of episode scores is empty")
    num_agents = len(episode_scores[0])

    plt.figure(1)
    scores_by_agent = {
        i: [episode_score[i] for episode_score in episode_scores]
        for i in range(num_agents)
    }
    scores_t_by_agent = {
        i: torch.tensor(scores, dtype=torch.float)
        for i, scores in scores_by_agent.items()
    }
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    for i in range(num_agents):
        scores_t = scores_t_by_agent[i]
        plt.plot(scores_t.numpy())
        # Take 100 episode averages and plot them too
        if len(scores_t) >= 100:
            means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
