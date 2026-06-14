from src.rl.environment import ForexTradingEnv
from src.utils.config import RLConfig
import numpy as np

cfg = RLConfig()
X = np.random.randn(5000, 60, 32).astype(np.float32)
P = np.ones((5000, 2), dtype=np.float32) * 1.1
env = ForexTradingEnv(X, P, cfg)
obs, info = env.reset()
print("obs shape:", obs.shape)

# Test: buy at bar 0, hold 50 bars, sell at bar 50
# cooldown=30 so next entry at bar 80
# buy again at bar 80, hold 50 bars, sell at bar 130 => 2 trades in 130 bars
actions = [0]*200
actions[0]  = 1   # buy
actions[50] = 2   # sell (close long) -> cooldown 30 bars
actions[51] = 1   # buy blocked by cooldown
actions[60] = 1   # still blocked
actions[80] = 1   # buy again (cooldown expired at bar 80)
actions[130]= 2   # sell (close long)

for i, action in enumerate(actions):
    obs, rew, term, trunc, info = env.step(action)
    if action != 0:
        pos = info["position"]
        cd  = info["cooldown_remaining"]
        tr  = info["trades"]
        print(f"bar {i}: action={action}, pos={pos}, trades={tr}, cooldown={cd}, reward={rew:.3f}")
    if term or trunc:
        break

print()
print(f"Final trades: {info['trades']} (expected 2)")
print("PASSED" if info["trades"] == 2 else "FAILED")
