## Introduction

This is the official implementation of the article titled 'Optimizing Exploration and Exploitation in Deep Reinforcement Learning through State Gradient Utilization.' In this project, we implement the State Gradient Refinement Module (SGRM) to enhance exploration and exploitation in Deep Reinforcement Learning (DRL). The approach is demonstrated across various DRL algorithms, showcasing substantial performance improvements, especially in environments where states are represented using images.

## Prerequisites

- Python 3.8
- PyTorch 2.0.1

## Installation

To set up the project environment:

```bash
git clone https://github.com/zh4men9/SGRM.git
cd SGRM
pip install -r requirements.txt
```

## Usage

Here's a quick start on how to run the algorithms:

```python
python ./src/diff_mujoco_td3.py --task Ant-v3 --seed 0 --logger wandb --gradient_order 0
```

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contact

- Author: zh4men9
- Email: zh4men9@163.com