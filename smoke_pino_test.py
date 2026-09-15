import torch
from pinn_models import PINO, get_model

print('torch version:', torch.__version__)

# Instantiate PINO via class and via factory
m1 = PINO(None, hidden=32, layers=3)
m2 = get_model('pino')

for name, m in [('PINO_class', m1), ('PINO_factory', m2)]:
    m.train()
    x = torch.randn(8, 5, dtype=torch.float32)
    y = m(x)
    loss = (y ** 2).mean()
    loss.backward()
    print(f"{name}: y.shape={y.shape}, loss={loss.item():.6e}")
    print(f"{name}: y[0]={y[0].detach().cpu().numpy()}")

print('Smoke test complete')
