# Phase 3 — System Integration
## AI-Based Proactive Memory Optimization in Cloud Virtual Machines

This folder contains a complete, runnable implementation of the system integration layer:
your trained ensemble model (Phase 2) wired into a live KVM/libvirt environment via
virtio-balloon, with a feedback loop and a benchmark harness for the reactive-vs-proactive
comparison you'll need for Phase 4.

**Important:** this code must run on a Linux host with KVM/libvirt installed (a physical
machine or a VM with nested virtualization enabled — e.g. a bare-metal cloud instance, or
VirtualBox/VMware with nested virt turned on). It will NOT run in a sandboxed / restricted
container, since it needs `/dev/kvm` and libvirt daemon access.

---

## 1. Where to set this up

Pick one of these, in order of ease:

| Option | Notes |
|---|---|
| **Your own Linux laptop/desktop** (Ubuntu 22.04/24.04 recommended) | Easiest for a demo. Needs CPU virtualization support (Intel VT-x / AMD-V) enabled in BIOS. |
| **A bare-metal cloud instance** (e.g. AWS `.metal` instance, GCP with nested virt enabled, Azure `Dv3`/`Ev3` with nested virt) | Needed if your laptop doesn't support virtualization or you're on Apple Silicon / Windows without WSL2 KVM passthrough. |
| **A physical lab machine** at your institution | If PES gives you lab access with root, this is often simplest for a fixed demo setup. |

Do **not** try to do this inside Docker or inside another already-virtualized sandbox without
nested virtualization explicitly enabled — `/dev/kvm` won't exist and none of this will work.

Check virtualization support before anything else:

```bash
egrep -c '(vmx|svm)' /proc/cpuinfo   # >0 means your CPU supports it
kvm-ok                               # after installing cpu-checker (see below)
```

---

## 2. Host setup (run once)

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils \
    virtinst virt-manager cpu-checker python3-pip python3-venv stress-ng

# Add yourself to the libvirt/kvm groups so you don't need sudo for every command
sudo usermod -aG libvirt,kvm $USER
newgrp libvirt   # or log out and back in

# Verify
kvm-ok
sudo systemctl enable --now libvirtd
virsh list --all
```

---

## 3. Create the guest VM

```bash
# Download a small cloud image (Ubuntu Server minimal) to test with
wget https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img \
    -O /var/lib/libvirt/images/testvm.qcow2

sudo qemu-img resize /var/lib/libvirt/images/testvm.qcow2 10G

# Create the VM — note --memballoon virtio, this is the critical flag
sudo virt-install \
  --name testvm \
  --ram 2048 \
  --vcpus 2 \
  --disk path=/var/lib/libvirt/images/testvm.qcow2,format=qcow2 \
  --os-variant ubuntu22.04 \
  --network network=default \
  --graphics none \
  --import \
  --memballoon model=virtio
```

Confirm the balloon device exists:

```bash
virsh dumpxml testvm | grep -A2 memballoon
# <memballoon model='virtio'>
```

---

## 4. Guest-side setup (inside the VM)

SSH into the VM (or use `virsh console testvm`), then:

```bash
sudo apt update
sudo apt install -y qemu-guest-agent
sudo systemctl enable --now qemu-guest-agent
```

Without the guest agent, `virDomainMemoryStats` gives you host-side ballooning stats only —
you lose `unused`, `available`, and `usable`, which matter a lot for your feature vector.

Verify from the **host**:

```bash
virsh qemu-agent-command testvm '{"execute":"guest-ping"}'
# should return {"return":{}}
```

---

## 5. Python environment (on the host)

```bash
cd phase3_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Drop in your Phase 2 model

Copy your trained Voting Classifier (saved with `joblib.dump`) into this folder as:

```
phase3_system/models/ensemble_model.pkl
```

If you haven't saved it yet, in your training notebook:

```python
import joblib
joblib.dump(voting_clf, "ensemble_model.pkl")
```

Also save the scaler/encoder if you used one — `model_wrapper.py` expects
`models/scaler.pkl` too (skip this if you didn't scale features; see comments in that file).

---

## 7. File overview

| File | Role |
|---|---|
| `config.py` | All tunable constants — thresholds, memory limits, poll interval |
| `collector.py` | Polls libvirt for live memory stats from the guest |
| `features.py` | Turns raw stats history into the feature vector your model expects |
| `model_wrapper.py` | Loads the pickled ensemble, runs inference |
| `decision_engine.py` | Turns a prediction into an action, applies safety guardrails |
| `actuator.py` | Calls `setMemory()` on the domain (the actual balloon resize) |
| `feedback_logger.py` | Logs every action + outcome to CSV, for Phase 4 analysis / retraining |
| `orchestrator.py` | Main loop — wires everything together, run this to start the system |
| `benchmark.py` | Runs the reactive-vs-proactive comparison using `stress-ng` |

---

## 8. Running it

**Dry run first** (no actual memory changes, just logs what it *would* do):

```bash
python3 orchestrator.py --domain testvm --dry-run
```

**Live run:**

```bash
python3 orchestrator.py --domain testvm
```

**Generate memory pressure in another terminal** (SSH into the VM or `virsh console`):

```bash
stress-ng --vm 2 --vm-bytes 80% --timeout 120s
```

Watch `orchestrator.py`'s output — you should see stats being polled, predictions being made,
and (if triggered) `setMemory()` calls with before/after values.

**Run the benchmark** (produces the comparison data for your Phase 4 report):

```bash
python3 benchmark.py --domain testvm --modes none reactive proactive --duration 120
```

This writes `benchmark_results.csv` with per-mode latency/swap/recovery-time stats.

---

## 9. Common issues

| Symptom | Fix |
|---|---|
| `libvirt.libvirtError: Failed to connect to the hypervisor` | Run `sudo systemctl status libvirtd`; make sure you're in the `libvirt` group and re-logged in |
| `memoryStats()` returns empty dict | Guest agent not running/installed inside the VM — see step 4 |
| `setMemory()` has no visible effect inside guest | Guest kernel doesn't have `virtio_balloon` loaded — check `lsmod \| grep balloon` inside the VM |
| Model predicts constantly "increase" | Check your feature vector units/scale match training — a mismatched scaler is the #1 cause |
| `stress-ng: command not found` inside guest | Install it inside the guest too, not just the host, if you want in-guest pressure |

---

## 10. What to show in your Phase 3 demo

1. `orchestrator.py` running live, polling stats every few seconds
2. Trigger `stress-ng` inside the guest → show the model firing a prediction → show
   `virsh dominfo testvm` memory value changing in real time
3. `benchmark.py` output comparing no-management vs reactive vs your proactive system
   (this is your strongest slide — quantifies the "proactive" claim from your abstract)
