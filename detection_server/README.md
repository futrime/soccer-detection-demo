# Detection Server

The detection server of the demo

## Install

HuggingFace is blocked in China. You can run the following command or add it to your `.bashrc` to configure a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

This project is composed of a detection server and a ROS node. For the server, it requires Python 3.12 environment. We recommend using Miniconda to create a new environment.

First, create a new conda environment:

```bash
conda create -n soccer python=3.12
```

Since `PYTHONPATH` is set by ROS, you need to unset it before running the server:

```bash
unset PYTHONPATH
```

Then, activate the environment and install the dependencies:

```bash
conda activate soccer
pip install -r requirements.txt
```

## Usage

First, activate the environment:

```bash
conda activate soccer
```

Then, run the server:

```bash
python main.py
```

## Contributing

PRs are welcome!

## License

MIT © Zijian Zhang
