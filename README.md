# Physics-Aware Platoon IDS

Physics-aware intrusion detection system (IDS) for longitudinal truck platooning using V2V and V2I communications.

This project focuses on detecting false data injection (FDI) attacks in connected truck platoons through:
- spatiotemporal feature engineering,
- cyber-physical consistency analysis,
- sequence learning,
- and realistic highway driving behavior.

## Project Objectives

- Model realistic longitudinal truck platooning dynamics
- Simulate V2V and RSU-assisted V2I communication
- Generate realistic false data injection attacks
- Build a lightweight but robust IDS architecture
- Use highway trajectory datasets (highD) for realistic calibration

## Current Research Focus

- Longitudinal truck platooning
- Physics-aware consistency features
- BiLSTM + attention architectures
- V2V BSM manipulation
- RSU guidance message integration
- Sliding temporal windows
- Realistic acceleration profile generation

## Dataset

This project uses the highD highway trajectory dataset:

https://levelxdata.com/highd-dataset/

The dataset itself is NOT included in this repository.

## Repository Structure

```text
data/
simulation/
models/
notebooks/
results/
docs/
