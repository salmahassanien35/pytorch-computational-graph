import torch

x = torch.tensor(1.0, requires_grad=True)
print("Input x =", x.item())

w00 = torch.tensor(0.5, requires_grad=True)
b00 = torch.tensor(0.1, requires_grad=True)
w01 = torch.tensor(-0.3, requires_grad=True)
b01 = torch.tensor(0.2, requires_grad=True)
w02 = torch.tensor(0.8, requires_grad=True)
b02 = torch.tensor(-0.5, requires_grad=True)

z00 = w00 * x + b00
z01 = w01 * x + b01
z02 = w02 * x + b02

a00 = torch.relu(z00)
a01 = torch.relu(z01)
a02 = torch.relu(z02)

print("\nLayer 1 outputs:")
print(f"a00={a00.item():.4f}, a01={a01.item():.4f}, a02={a02.item():.4f}")

# 3️⃣ Layer 2 (2 neurons) → Activation: Sigmoid
w10 = torch.tensor(0.6, requires_grad=True)
b10 = torch.tensor(0.1, requires_grad=True)
w11 = torch.tensor(-0.4, requires_grad=True)
b11 = torch.tensor(0.05, requires_grad=True)

layer1_sum = a00 + a01 + a02

z10 = w10 * layer1_sum + b10
z11 = w11 * layer1_sum + b11

a10 = torch.sigmoid(z10)
a11 = torch.sigmoid(z11)

print("\nLayer 2 outputs:")
print(f"a10={a10.item():.4f}, a11={a11.item():.4f}")

combined = a10 + a11
a_combined = torch.tanh(combined)

print("\nCombined output after Tanh =", a_combined.item())

w20 = torch.tensor(1.2, requires_grad=True)
b20 = torch.tensor(-0.3, requires_grad=True)

output = w20 * a_combined + b20
print("\nFinal output =", output.item())

output.backward()

print("\nGradient of output w.r.t input x =", x.grad.item())
