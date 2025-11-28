import gymnasium as gym
from ppo_agent import PPO_Agent

def main():
    # Inicializar entorno
    env = gym.make("Pendulum-v1")
    
    # Obtener dimensiones del entorno
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Hiperparámetros básicos
    N_EPISODES = 1000
    LR = 3e-4
    
    # Inicializar Agente PPO
    agent = PPO_Agent(state_dim, action_dim, lr=LR)
    
    print(f"Entorno: Pendulum-v1")
    print(f"Dimensiones Estado: {state_dim}, Acción: {action_dim}")
    print("Iniciando bucle de entrenamiento...")

    # Bucle de entrenamiento
    for episode in range(N_EPISODES):
        # Resetear entorno
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        # Aquí iría el bucle de interacción con el entorno (rollout)
        # while not (done or truncated):
        #     action = agent.select_action(state) ...
        #     next_state, reward, ... = env.step(action)
        #     agent.store_transition(...)
        #     state = next_state
        
        # Actualizar agente
        # agent.update()
        
        pass

if __name__ == "__main__":
    main()
