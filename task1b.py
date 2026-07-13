"""
===================================================
    eLSI Sprint 1 - Task 1B : Q-Learning
===================================================

Participant template.

HOW TO RUN
  1. Open the Task 1B scene in CoppeliaSim.
  2. Start the bridge:   python3 bridge_task1b.py --eval
  3. Train:              python3 task1b_template.py --mode train
     Test (no learning): python3 task1b_template.py --mode test

MODES
  train : choose actions with exploration AND update the Q-table.
          The Q-table is saved to disk on exit.
  test  : load the saved Q-table, act greedily, and DO NOT update it.

WHAT YOU IMPLEMENT
  get_state()     - how to turn the 5 sensor values into a discrete state.
  get_reward()    - how good the latest reading is.
  choose_action() - which action to take in a given state (the policy).

Team ID: [ 685 ]

-------------------------------------------------------------------------
FIXES APPLIED (curves / sharp turns were failing) - see inline comments
marked with [FIX n] for the reasoning behind each change.
-------------------------------------------------------------------------
"""

import time
import os
import pickle
import random
import argparse

from connector_task1b import CoppeliaClient

# The five line sensors, ordered left -> right across the robot ([0.0, 1.0]).
SENSOR_ORDER = ['left_corner', 'left', 'middle', 'right', 'right_corner']

# [FIX 9] Sensor POLARITY depends on the track: line sensors report
# reflectance, so a WHITE line on a BLACK ground gives HIGH readings on the
# line, while a BLACK line on a WHITE ground gives LOW readings on the line.
# get_state()/get_reward() below are written assuming "line = high reading".
# Set this flag to match whichever track you're running so that assumption
# holds in both cases - everything downstream is unaffected.
#
#   "white_on_black" : white line, black ground (line reads HIGH)   <- default
#   "black_on_white" : black line, white ground  (line reads LOW)

# [FIX 1] Action set now has a genuine pivot-strength turn ("hard left/right")
# between the old "sharp" turn and the full in-place pivot used for recovery.
# This gives the agent something to reach for on a real corner instead of a
# big jump straight from a mild differential to a 180-style spin.
ACTIONS = [
    (4.0, 4.0),    # 0: forward

    (3.2, 4.0),    # 1: left
    (4.0, 3.2),    # 2: right

    (2.2, 4.0),    # 3: sharp left
    (4.0, 2.2),    # 4: sharp right

    (0.2, 4.0),    # 5: hard left  (near-pivot, still rolling forward a bit)
    (4.0, 0.2),    # 6: hard right

    (-1.5, 1.5),   # 7: spin left  (recovery / lost)
    (1.5, -1.5),   # 8: spin right
]

# Hyper parameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 0.2     # [FIX 2] raised from 0.1 - more exploration needed for
                         # the new, larger state space (see get_state below).
EPSILON_END = 0.1       # [FIX 2] allow epsilon to actually decay down, instead
                         # of START == END which made decay a no-op before.
EPSILON_DECAY = 0.995


Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.pkl")

# =============================================================================
#  get_state()
# =============================================================================
def get_state(sensors): 
 global previous_state
 vals = [sensors[s] for s in SENSOR_ORDER]

# --------- Auto detect polarity ---------

 median = sorted(vals)[2]
 diffs = [v - median for v in vals]

 strongest = max(diffs, key=abs)

 if abs(strongest) < 0.02:
    return ("LOST",)

 line_positive = strongest > 0

# If line is dark, invert values
 if not line_positive:
    vals = [1.0 - v for v in vals]

# --------- Centroid ---------

 total = sum(vals)

 if total < 0.05:
    return ("LOST",)

 positions = [-2, -1, 0, 1, 2]

 centroid = sum(p * v for p, v in zip(positions, vals)) / total

 mag = abs(centroid)

 if mag < 0.25:
    state = ("CENTER",)

 elif mag < 0.60:
    side = "RIGHT" if centroid > 0 else "LEFT"
    state = (side, 1)

 elif mag < 1.0:
    side = "RIGHT" if centroid > 0 else "LEFT"
    state = (("FAR_" + side), 2)

 else:
    side = "RIGHT" if centroid > 0 else "LEFT"
    state = (("FAR_" + side), 3)

 previous_state = state
 return state

# =============================================================================
#  get_reward()
# =============================================================================
def get_reward(sensors, state):
    """
    [FIX 4] The old reward gave FAR_LEFT/FAR_RIGHT almost nothing (+1),
    barely above LOST (-20). On a sustained curve the robot legitimately
    needs to SIT in FAR_LEFT/FAR_RIGHT for many consecutive steps - punishing
    that state taught the agent to snap back toward CENTER mid-curve,
    causing the overshoot/oscillation you'd see on sharp turns.

    Fix: reward is now based on "is the line still under the sensor array
    at all", not "how close to dead-center is it". Being correctly locked
    onto a sharp curve is now rewarded close to CENTER, not punished.
    """
    s = state[0]

    if s == "CENTER":
        return 20

    if s in ("LEFT", "RIGHT"):
        # intensity 1 = mild correction, fine; deeper intensity here just
        # means a real curve is starting - still good tracking.
        return 15

    if s in ("FAR_LEFT", "FAR_RIGHT"):
        intensity = state[1] if len(state) > 1 else 1
        if intensity >= 2:
            # robot is correctly hugging a sharp turn - reward this well,
            # don't punish it for not being centered.
            return 12
        return 8

    return -20  # LOST


# =============================================================================
#  choose_action()
# =============================================================================
def choose_action(agent, state, training):
    """
    Epsilon-greedy policy. [FIX 5] No behavioural change needed here once the
    state space and bootstrap table carry the intensity information - this
    function was already correct, the problem was upstream (state/reward).
    """
    agent._ensure(state)

    if training and random.random() < agent.epsilon:
        return random.randint(0, agent.n_actions - 1)

    best_q = max(agent.q_table[state])
    best_actions = [i for i, q in enumerate(agent.q_table[state]) if q == best_q]
    return random.choice(best_actions)


# =============================================================================
#  Q-learning agent (do not edit)
# =============================================================================
class QLearningAgent:
    def __init__(self, n_actions, alpha, gamma, epsilon, path):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.path = path
        self.q_table = {}

    def _ensure(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions

    def update(self, state, action, reward, next_state):
        self._ensure(state)
        self._ensure(next_state)
        best_next = max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Loaded Q-table ({len(self.q_table)} states) from {self.path}")
            return True
        return False

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Saved Q-table ({len(self.q_table)} states) to {self.path}")


# =============================================================================
#  Main loop
# =============================================================================
def run(mode):
    training = (mode == "train")

    agent = QLearningAgent(len(ACTIONS), ALPHA, GAMMA, EPSILON_START, Q_TABLE_PATH)
    loaded = agent.load()

    # [FIX 6] Bootstrap table updated for the new (side, intensity) states and
    # the new action indices (0-8, including the new hard-left/hard-right at
    # 5/6 and recovery spins moved to 7/8). Intensity-2/3 states are seeded
    # toward the stronger "hard" turn, not the old mild "sharp" turn, so the
    # agent starts with a sane prior for actual corners instead of learning
    # it from scratch via random exploration alone.
    bootstrap = {
        ("CENTER",):           [20, 0, 0, 0, 0, 0, 0, 0, 0],

        ("RIGHT", 1):          [5, 0, 12, 0, 8, 0, 0, 0, 0],
        ("RIGHT", 2):          [0, 0, 8, 0, 14, 0, 4, 0, 0],
        ("RIGHT", 3):          [0, 0, 4, 0, 10, 0, 14, 0, 0],
        ("FAR_RIGHT", 1):      [0, 0, 8, 0, 16, 0, 2, 0, 0],
        ("FAR_RIGHT", 2):      [0, 0, 2, 0, 10, 0, 16, 0, 0],
        ("FAR_RIGHT", 3):      [0, 0, 0, 0, 12, 0, 18, 0, 0],

        ("LEFT", 1):           [5, 12, 0, 8, 0, 0, 0, 0, 0],
        ("LEFT", 2):           [0, 8, 0, 14, 0, 4, 0, 0, 0],
        ("LEFT", 3):           [0, 4, 0, 10, 0, 14, 0, 0, 0],
        ("FAR_LEFT", 1):       [0, 16, 0, 8, 0, 2, 0, 0, 0],
        ("FAR_LEFT", 2):       [0, 10, 0, 2, 0, 16, 0, 0, 0],
        ("FAR_LEFT", 3):       [0, 0, 0, 12, 0, 18, 0, 0, 0],

        ("LOST",):             [0, 0, 0, 0, 0, 0, 0, 10, 10],
    }
    for s, q in bootstrap.items():
        if s not in agent.q_table:
            agent.q_table[s] = q[:]

    if loaded and training:
        agent.epsilon = EPSILON_START  # reset exploration if we're training, even with a loaded table
    if not training and not loaded:
        print("ERROR: test mode needs a trained Q-table. Run --mode train first.")
        return

    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print(f"Connected to bridge_task1b. Mode = {mode}. (Ctrl+C to stop)")

    prev_state = None
    prev_action = None
    reward = 0.0

    try:
        consecutive_none = 0
        step_count = 0
        last_line_side = None
        lost_steps = 0

        while True:
            sensors = client.receive_sensor_data()

            if sensors is None:
                consecutive_none += 1
                if consecutive_none > 20:
                    print("Sensor dropout detected, resetting state tracking...")
                    consecutive_none = 0
                    prev_state = None
                    prev_action = None
                time.sleep(0.02)
                continue
            consecutive_none = 0

            state = get_state(sensors)
            reward = get_reward(sensors, state)
            state_name = state[0]

            if state_name in ("LEFT", "FAR_LEFT"):
                last_line_side = "left"
            elif state_name in ("RIGHT", "FAR_RIGHT"):
                last_line_side = "right"

            if state_name == "LOST":
                lost_steps += 1
            else:
                lost_steps = 0

            # [FIX 7] Recovery is now ONLY a hardcoded spin once we are
            # *genuinely* lost for a while (line fully off all 5 sensors for
            # many consecutive steps). It is no longer triggered by being
            # deep in a turn, because deep-in-a-turn is now its own
            # well-defined, well-rewarded state (FAR_LEFT/FAR_RIGHT,
            # intensity 2/3) that the agent actually learns to handle via
            # the Q-table - it is NOT lost, so it should NOT be hijacked.
            is_recovery = (lost_steps > 15 and last_line_side is not None)

            # [FIX 8] The old code blocked learning whenever lost_steps != 0,
            # which - combined with the old bug above - meant the agent
            # never got to learn from the exact moments a sharp turn
            # temporarily looked like "lost". Now we only withhold learning
            # during genuine hardcoded recovery, so every real turn step
            # (lost_steps == 0, which is most turn frames with the new
            # wider sensor-based classification) is learned from normally.
            if training and prev_state is not None and not is_recovery:
                agent.update(prev_state, prev_action, reward, state)

            if training and step_count % 50 == 0:
                agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

            if is_recovery:
                action = 7 if last_line_side == 'left' else 8
            else:
                action = choose_action(agent, state, training)

            left, right = ACTIONS[action]
            client.send_motor_command(left, right, state=list(state), reward=reward, action=action)

            if not is_recovery:
                prev_state, prev_action = state, action
            step_count += 1
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            client.send_motor_command(0.0, 0.0, state=0, reward=0.0, action=0)
        except Exception:
            pass
        client.close()
        if training:
            agent.save()


def main():
    parser = argparse.ArgumentParser(description="Task 1B - Q-Learning")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="train: explore + update Q-table; test: greedy, no update")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()