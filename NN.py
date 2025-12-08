import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE DATOS (El "Problema" a resolver)
# ---------------------------------------------------------
# Imaginemos que la función real es y = 2x + 4 (W=2, b=4)
# Generamos algunos datos sintéticos basados en esa función
X = np.array([-2, -1, 1, 2], dtype=float)
Y_real = np.array([6, 2, -2, -4], dtype=float)

# ---------------------------------------------------- -----
# 2. INICIALIZACIÓN DE LA RED (Los "Intentos")
# ---------------------------------------------------------
# Empezamos con valores aleatorios (la red no sabe nada aún)
w = 10  # Peso inicial (pendiente)
b = 10  # Sesgo inicial (intersección Y)

# ---------------------------------------------------------
# 3. HIPERPARÁMETROS (La configuración del entrenamiento)
# ---------------------------------------------------------
learning_rate = 0.03 # Qué tan grandes son los pasos que damos
epochs = 550        # Cuántas veces repetimos el proceso

loss_history = []     # Para guardar el historial de errores

print(f"Inicio -> w: {w}, b: {b}")

# ---------------------------------------------------------
# 4. BUCLE DE ENTRENAMIENTO (El "Gimnasio")
# ---------------------------------------------------------
for epoch in range(epochs):
    
    # A. FORWARD PASS (Predicción)
    # Calculamos qué opina la red con los pesos actuales
    y_pred = (w * X) + b
    
    # B. CALCULAR PÉRDIDA (Loss - MSE)
    # Error = (Valor Real - Predicción)
    error = Y_real - y_pred
    # Loss = Promedio de los errores al cuadrado
    loss = np.mean(error ** 2)
    loss_history.append(loss)
    
    # C. BACKPROPAGATION (Calcular Gradientes)
    # Estas fórmulas vienen de derivar la función de pérdida.
    # Nos dicen la dirección en la que debemos mover w y b.
    
    # Derivada con respecto a w: -2 * promedio(x * error)
    w_gradient = -2 * np.mean(X * error)
    
    # Derivada con respecto a b: -2 * promedio(error)
    b_gradient = -2 * np.mean(error)
    
    # D. OPTIMIZACIÓN (Actualizar pesos)
    # Nos movemos en dirección opuesta al gradiente (hacia abajo en la montaña)
    w = w - (learning_rate * w_gradient)
    b = b - (learning_rate * b_gradient)

    # Imprimir progreso cada 50 épocas
    if epoch % 50 == 0:
        print(f"Época {epoch}: Loss: {loss:.4f} | w: {w:.4f}, b: {b:.4f}")

# ---------------------------------------------------------
# 5. RESULTADOS Y VISUALIZACIÓN
# ---------------------------------------------------------
print("-" * 30)
print(f"ENTRENAMIENTO FINALIZADO.")
print(f"Valores encontrados -> w: {w:.4f}, b: {b:.4f}")
print(f"Valores reales      -> w: 2.0000, b: 4.0000")

# Gráfica
plt.figure(figsize=(10, 5))

# Subgráfico 1: Ajuste de la línea
plt.subplot(1, 2, 1)
plt.scatter(X, Y_real, color='blue', label='Datos Reales')
plt.plot(X, (w * X) + b, color='red', label='Predicción de la Red')
plt.title('Ajuste de Regresión Lineal')
plt.legend()

# Subgráfico 2: Descenso del error
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title('Historial de Pérdida (Loss)')
plt.xlabel('Época')
plt.ylabel('Error')

plt.tight_layout()
plt.show()