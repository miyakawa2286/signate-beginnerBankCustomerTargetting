FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN apt-get update

RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:22861238' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
RUN sed -i 's/#Port 22/Port 20022/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

EXPOSE 20022
CMD ["/usr/sbin/sshd", "-D"]

RUN apt-get install -y unzip

# update pip
RUN pip install --upgrade pip setuptools wheel
# remove conda
RUN pip uninstall -y $(pip list | grep conda)

# intall kaggle api
COPY ./kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json
RUN pip install kaggle

# install requirements
COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
