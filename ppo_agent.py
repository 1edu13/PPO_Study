import torch
import torch.nn as nn
import numpy as np

# --- EL PILOTO (Red Neuronal) ---
class CarPilot(nn.Module):
    def __init__(self, num_sensors, num_actions):
        super(CarPilot, self).__init__()
        # Capas compartidas
        self.shared_layers = nn.Sequential(
            nn.Linear(num_sensors, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Actor (Media de la acción)
        self.actor_mean = nn.Linear(64, num_actions)
        # Desviación estándar aprendible
        self.actor_log_std = nn.Parameter(torch.zeros(1, num_actions))
        # Crítico (Valor del estado)
        self.critic = nn.Linear(64, 1)

    def get_action(self, state, action=None):
        z = self.shared_layers(state)
        mean = self.actor_mean(z)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        # Sumamos log_prob en la última dimensión para manejar acciones multidimensionales
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(z)
        
        return action, log_prob, entropy, value

# --- EL ENTRENADOR PPO (Algoritmo de Aprendizaje) ---
class PPO_Agent:
    def __init__(self, num_sensors, num_actions, lr, gamma, K_epochs, eps_clip):
        # Hiperparámetros
        self.gamma = gamma          # Factor de descuento (futuro vs presente)
        self.eps_clip = eps_clip    # Rango de recorte (0.1 - 0.2) para no cambiar bruscamente
        self.K_epochs = K_epochs    # Cuántas veces repasamos los mismos datos

        # Inicializamos el Cerebro (Policy)
        self.policy = CarPilot(num_sensors, num_actions)
        
        # El Optimizador (El mecánico que ajusta los pesos)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Función de pérdida para el Crítico (Error Cuadrático Medio)
        self.MseLoss = nn.MSELoss()

        # MEMORIA (Buffer)
        # Aquí guardaremos lo que pasa en cada episodio
        self.buffer = []

    def store_data(self, transition):
        """Guardamos una experiencia: (estado, accion, log_prob, reward, done)"""
        self.buffer.append(transition)

    def update(self):
        """
        ¡LA MAGIA DEL PPO! 
        Aquí es donde el agente reflexiona sobre lo que hizo y aprende.
        """
        # 1. PREPARAR LOS DATOS
        # Convertimos la lista de memoria en tensores de PyTorch para calcular rápido
        states      = torch.tensor(np.array([t[0] for t in self.buffer]), dtype=torch.float)
        actions     = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.float)
        old_log_probs = torch.tensor(np.array([t[2] for t in self.buffer]), dtype=torch.float)
        rewards     = [t[3] for t in self.buffer]
        dones       = [t[4] for t in self.buffer]

        # 2. CALCULAR EL RETORNO (Monte Carlo)
        # Calculamos cuánto "premio real" ganó en cada paso mirando hacia el futuro
        returns = []
        discounted_sum = 0
        for reward, is_done in zip(reversed(rewards), reversed(dones)):
            if is_done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        
        # Normalizamos los retornos (ayuda a que el entrenamiento sea estable)
        returns = torch.tensor(returns, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # 3. BUCLE DE OPTIMIZACIÓN (K Epochs)
        # Repasamos los datos varias veces para aprender bien
        for _ in range(self.K_epochs):
            
            # Pasamos los estados viejos por la red ACTUAL para ver qué opina ahora
            # (action=actions fuerza a la red a evaluar las acciones que ya tomamos)
            _, log_probs, entropy, state_values = self.policy.get_action(states, actions)
            
            # Calculamos la VENTAJA (Advantage)
            # ¿Qué tan bueno fue el resultado comparado con lo que esperaba el crítico?
            state_values = torch.squeeze(state_values)
            advantages = returns - state_values.detach()

            # --- FÓRMULA DEL RATIO (Importance Sampling) ---
            # Comparamos la probabilidad nueva vs la vieja
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # --- FÓRMULA DEL CLIPPING (El Freno de Seguridad) ---
            # Si el cambio es muy brusco, lo recortamos
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Loss Total = -min(surr) + Error del Crítico - Bonificación por Exploración (Entropía)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * entropy

            # 4. ACTUALIZAR LA RED (Backpropagation)
            self.optimizer.zero_grad() # Limpiamos residuos anteriores
            loss.mean().backward()     # Calculamos gradientes
            self.optimizer.step()      # Aplicamos los cambios

        # 5. LIMPIAR MEMORIA (On-Policy)
        # PPO no reutiliza datos viejos, borramos todo para la siguiente ronda
        self.buffer = []