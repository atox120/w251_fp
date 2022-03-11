# Readme

You can build the dockerfile locally by running:

```bash
sudo docker build -f Dockerfile_server -t atox120/w251_fp_server:<version_tag> .
```
with an appropriate version tag. Or else it can be pulled directly from dockerhub:

```bash
sudo docker pull nvidia w251_fp_server:latest
```
It can be run with a command like:
```ba
sudo docker run --rm -it --ipc=host --net=host --gpus=all -v /path/to/dir/:/workspace/w251 --runtime nvidia w251_fp_server:latest
```
setting the appropriate file path. 

## Running a demo

Within the container, you can start up a notebook:
```python
jupyter-lab --allow-root
```
Within the Video-Swin-Transformer folder there is a demo notebook named 'AT_DEMO.ipynb'. Open this demo and run the cell.
Note there are other demo's in the 'demo' folder of this directory. They can be run with the appropriate checkpoint folder set up and model checkpoint downloaded. See the repository for more information. 
