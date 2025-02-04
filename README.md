```bash

docker run -ti --rm \
-v ~/work/code/py_code/hylee/ml:/src \
-w /src \
docker-mirrors.alauda.cn/library/python:3.10.12-bullseye \
bash

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "black[jupyter]"

docker run -it --rm -p 10000:8888 \
-v ~/work/code/py_code/hylee/ml:/home/jovyan/work \
jupyter/minimal-notebook:x86_64-python-3.11.6




```
