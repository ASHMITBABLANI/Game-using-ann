import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pong-secret-key!'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- GAME & CANVAS DIMENSIONS ---
WIDTH, HEIGHT = 800, 500
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 90
BALL_SIZE = 15
PADDLE_SPEED = 7

# --- NEURAL NETWORK ARCHITECTURE ---
class PongANN(nn.Module):
    def __init__(self):
        super(PongANN, self).__init__()
        # Input Layer: 5 features
        # Hidden Layer 1: 16 neurons
        # Hidden Layer 2: 8 neurons
        # Output Layer: 3 actions (0: Stay, 1: Up, 2: Down)
        self.fc1 = nn.Linear(5, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward_with_activations(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out, h1, h2

# Train model on startup
def pretrain_model():
    model = PongANN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    states, labels = [], []
    for _ in range(15000):
        ai_y = random.uniform(0, HEIGHT - PADDLE_HEIGHT)
        ball_x = random.uniform(WIDTH // 2, WIDTH)
        ball_y = random.uniform(0, HEIGHT)
        ball_vx = random.uniform(5, 10)
        ball_vy = random.uniform(-8, 8)

        paddle_center = ai_y + PADDLE_HEIGHT / 2.0
        if abs(paddle_center - ball_y) < 15:
            action = 0
        elif paddle_center > ball_y:
            action = 1
        else:
            action = 2

        states.append([ai_y / HEIGHT, ball_x / WIDTH, ball_y / HEIGHT, ball_vx / 10.0, ball_vy / 10.0])
        labels.append(action)

    X_train = torch.tensor(states, dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.long)

    for epoch in range(100):
        optimizer.zero_grad()
        out, _, _ = model.forward_with_activations(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    return model

ann_model = pretrain_model()

# --- GAME STATE ---
game_state = {
    'human_y': HEIGHT // 2 - PADDLE_HEIGHT // 2,
    'ai_y': HEIGHT // 2 - PADDLE_HEIGHT // 2,
    'ball_x': WIDTH // 2 - BALL_SIZE // 2,
    'ball_y': HEIGHT // 2 - BALL_SIZE // 2,
    'ball_vx': 6,
    'ball_vy': 4,
    'human_score': 0,
    'ai_score': 0
}

def reset_ball():
    game_state['ball_x'] = WIDTH // 2 - BALL_SIZE // 2
    game_state['ball_y'] = HEIGHT // 2 - BALL_SIZE // 2
    game_state['ball_vx'] = random.choice([-6, 6])
    game_state['ball_vy'] = random.choice([-4, 4])

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('player_mouse')
def handle_player_mouse(data):
    # Position paddle center directly on mouse Y position
    target_y = data['mouse_y'] - (PADDLE_HEIGHT / 2)
    # Constrain within canvas height
    game_state['human_y'] = max(0, min(HEIGHT - PADDLE_HEIGHT, target_y))

def game_loop():
    while True:
        time.sleep(1 / 60.0)  # 60 FPS tick rate

        # 1. ANN Inference for AI Paddle
        inputs = [
            game_state['ai_y'] / HEIGHT,
            game_state['ball_x'] / WIDTH,
            game_state['ball_y'] / HEIGHT,
            game_state['ball_vx'] / 10.0,
            game_state['ball_vy'] / 10.0
        ]
        
        with torch.no_grad():
            inp_t = torch.tensor([inputs], dtype=torch.float32)
            logits, h1, h2 = ann_model.forward_with_activations(inp_t)
            ai_action = torch.argmax(logits, dim=1).item()

        # Execute AI Action
        if ai_action == 1 and game_state['ai_y'] > 0:
            game_state['ai_y'] -= PADDLE_SPEED
        elif ai_action == 2 and game_state['ai_y'] < HEIGHT - PADDLE_HEIGHT:
            game_state['ai_y'] += PADDLE_SPEED

        # 2. Update Ball Physics
        game_state['ball_x'] += game_state['ball_vx']
        game_state['ball_y'] += game_state['ball_vy']

        # Wall Bounces
        if game_state['ball_y'] <= 0 or game_state['ball_y'] >= HEIGHT - BALL_SIZE:
            game_state['ball_vy'] *= -1

        # Left Paddle Bounce (Human)
        if (game_state['ball_x'] <= 30 + PADDLE_WIDTH and
            game_state['ball_x'] >= 30 and
            game_state['ball_y'] + BALL_SIZE >= game_state['human_y'] and
            game_state['ball_y'] <= game_state['human_y'] + PADDLE_HEIGHT and
            game_state['ball_vx'] < 0):
            game_state['ball_vx'] *= -1.05

        # Right Paddle Bounce (ANN AI)
        if (game_state['ball_x'] + BALL_SIZE >= WIDTH - 30 - PADDLE_WIDTH and
            game_state['ball_x'] + BALL_SIZE <= WIDTH - 30 and
            game_state['ball_y'] + BALL_SIZE >= game_state['ai_y'] and
            game_state['ball_y'] <= game_state['ai_y'] + PADDLE_HEIGHT and
            game_state['ball_vx'] > 0):
            game_state['ball_vx'] *= -1.05

        # Scoring Logic
        if game_state['ball_x'] < 0:
            game_state['ai_score'] += 1
            reset_ball()
        elif game_state['ball_x'] > WIDTH:
            game_state['human_score'] += 1
            reset_ball()

        # Broadcast state & activations to client
        socketio.emit('state_update', {
            'state': game_state,
            'activations': {
                'input': inputs,
                'h1': h1.numpy()[0].tolist(),
                'h2': h2.numpy()[0].tolist(),
                'output': logits.numpy()[0].tolist()
            }
        })

if __name__ == '__main__':
    socketio.start_background_task(game_loop)
    socketio.run(app, debug=True, port=5000)