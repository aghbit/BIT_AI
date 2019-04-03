Google Colaboratory link: https://colab.research.google.com/drive/1em2GBu4Gxbgt3OV79R9V2iwP6Yp8y3vx

Instrukcja:

Polecam zrobić to w Google Colaboratory, bo nie potrzebuje żadnego setup'u

Jak lubicie trudności, to:

Jeżeli macie w komputerze Nvidia GPU (i możecie znaleść swoją kartę tutaj https://developer.nvidia.com/cuda-gpus ) to możece po prostu zainstalować CUDA Toolkit 

https://developer.nvidia.com/cuda-80-ga2-download-archive

(Testowałem notebooka na cuda 8) 

W innym przypadku:
Polecam skorzystać z Google Cloud (albo innego cloud do którego macie dostęp i są tam karty NVIDIA) i pamęntajcie żeby zastopować VM jak skończycie. 

Osobiście mam VM na asia-east1-b bo są tam karty NVIDIA K80 które są tańsze godzinowo :) 

https://medium.com/@jayden.chua/quick-install-cuda-on-google-cloud-compute-6c85447f86a1

(Nie potrzebujecie cuDNN dla tej laby)

Jest to instrukcja jak skonfigurować VM dla cuda na Google Cloud

https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52

Jest to instrukcja jak zainstalować Jupyter notebook na Google Cloud 

Zeby skopiować file z computers na VM potrzebujecie polecenie:

gcloud compute scp /folder-to-your-file-locally/ instance-name:/tmp

Żeby skopiować notebook do domowej dyrektorii wystarczy wpisać:

sudo cp <nazwa_botebooka>.ipynb ~
