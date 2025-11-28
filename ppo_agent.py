import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# --- EL CEREBRO DEL COCHE (Red Neuronal) ---
class CarPilot(nn.Module):
    def __init__(self, num_sensors, num_actions):
        super(CarPilot, self).__init__()
        
        # 1. EL CUERPO (Procesa los sensores)
        # Recibe datos como velocidad, lidar, distancia a líneas...
        self.body = nn.Sequential(
            nn.Linear(num_sensors, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # 2. CABEZA DEL ACTOR (Decide el movimiento)
        # Calcula la media (mu) de la acción ideal (ej. giro de volante)
        self.actor_mean = nn.Linear(128, num_actions)
        
        # La desviación estándar (std) es aprendible. 
        # Empieza en 0 (e^0 = 1) para explorar, y la red aprenderá a reducirla.
        self.actor_log_std = nn.Parameter(torch.zeros(1, num_actions))

        # 3. CABEZA DEL CRÍTICO (Evalúa la situación)
        # Calcula cuánto "premio" esperamos ganar desde esta posición
        self.critic = nn.Linear(128, 1)

    def get_value(self, state):
        """
        Función para el Crítico: ¿Qué tan buena es esta situación?
        """
        features = self.body(state)
        return self.critic(features)

    def get_action(self, state, action=None):
        """
        Función para el Actor: ¡Conduce!
        Genera una distribución de probabilidad y saca una acción.
        """
        features = self.body(state)
        
        # Obtener estadísticas de la distribución
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std) # Convertimos log_std a número positivo
        
        # Crear la Distribución Normal (La "Nube" de posibles acciones)
        dist = Normal(mean, std)

        if action is None:
            # JUGANDO/EXPLORANDO:
            # Muestreamos una acción real de la distribución
            action = dist.sample()
        
        # CALCULANDO PROBABILIDADES (Para entrenar después):
        # Necesitamos saber qué tan probable era esta acción para las fórmulas del PPO
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1) # Mide la incertidumbre
        
        # Obtenemos también la opinión del crítico
        value = self.critic(features)

        return action, log_prob, entropy, value