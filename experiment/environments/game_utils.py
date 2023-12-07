import nashpy as nash

def get_game_rewards(game: nash.Game, actions: tuple[int, ...]):
    rewards = [
        game.payoff_matrices[agent_id][actions]
        for agent_id in range(len(game.payoff_matrices))
    ]
    return rewards
