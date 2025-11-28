import gymnasium as gym
import torch
from ppo_agent import PPO_Agent

def main():
    # --- 1. CONFIGURACIÓN DEL ENTORNO ---
    env_name = "Pendulum-v1"
    # render_mode="human" nos permite VER el péndulo. 
    # Si quieres que entrene súper rápido (sin verlo), cámbialo a None.
    env = gym.make(env_name, render_mode="human")

    # Obtenemos las dimensiones para configurar nuestro cerebro
    state_dim = env.observation_space.shape[0]  # 3 (cos, sin, vel)
    action_dim = env.action_space.shape[0]      # 1 (torque)

    # --- 2. HIPERPARÁMETROS (La "Receta" del aprendizaje) ---
    # Estos valores son estándar para PPO en problemas continuos
    lr = 0.0003           # Velocidad de aprendizaje (Learning Rate)
    gamma = 0.99          # Qué tanto le importa el futuro
    K_epochs = 10         # Cuántas veces repasa los datos en cada actualización
    eps_clip = 0.2        # Zona de confianza (20% de cambio máximo)
    
    # Frecuencia de actualización: ¿Cada cuántos pasos aprende?
    update_timestep = 2000 

    # --- 3. INICIALIZAR EL AGENTE ---
    # Aquí es donde INSTANCIAMOS la clase que creaste en el otro archivo
    agent = PPO_Agent(state_dim, action_dim, lr, gamma, K_epochs, eps_clip)

    # --- 4. BUCLE DE ENTRENAMIENTO ---
    max_episodes = 1000   # Cuántos juegos jugaremos
    time_step = 0         # Contador global de pasos

    print(f"--- Iniciando entrenamiento en {env_name} ---")

    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        current_ep_reward = 0

        # Bucle de un episodio (Jugar hasta que termine)
        while True:
            time_step += 1

            # A. EL CEREBRO DECIDE
            # Convertimos el estado a tensor porque PyTorch no entiende numpy
            state_tensor = torch.FloatTensor(state)
            
            # Le pedimos al agente una acción (y sus datos para aprender luego)
            # Nota: get_action devuelve tensores, usamos .detach().numpy() para pasarlo al entorno
            action, log_prob, _, _ = agent.policy.get_action(state_tensor)
            action = action.detach().numpy()
            log_prob = log_prob.detach() # No necesitamos gradientes aquí

            # B. ACTUAMOS EN EL MUNDO
            # El entorno espera la acción. En Pendulum, la acción va de -2 a 2.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # C. GUARDAMOS LA EXPERIENCIA
            # El agente necesita recordar qué pasó para aprender luego
            agent.store_data((state, action, log_prob, reward, done))

            # Actualizamos variables
            state = next_state
            current_ep_reward += reward

            # D. APRENDIZAJE (UPDATE)
            # Si hemos acumulado suficientes datos (2000 pasos), el agente para y aprende.
            if time_step % update_timestep == 0:
                print(f"🔄 Actualizando Agente (Paso {time_step})...")
                agent.update()

            # Si el episodio terminó, salimos del bucle while
            if done:
                break
        
        # Log del progreso
        print(f"Episodio {i_episode} \t Recompensa Total: {current_ep_reward:.2f}")

    env.close()

if __name__ == '__main__':
    main()