ARG BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04

###########################################
#  Base image
###########################################
FROM ${BASE_IMAGE} AS base
ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
	&& useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
	&& apt-get update \
	&& apt-get install -y sudo \
	&& echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
	&& chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip

# Install language
RUN apt-get update && apt-get install -y \
	locales \
	&& locale-gen en_US.UTF-8 \
	&& update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
	&& rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8

# Install timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
	&& export DEBIAN_FRONTEND=noninteractive \
	&& apt-get update \
	&& apt-get install -y tzdata \
	&& dpkg-reconfigure --frontend noninteractive tzdata \
	&& rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all
# Enable all capabilities: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.10.0/user-guide.html#driver-capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=all  
ENV QT_X11_NO_MITSHM=1

ENV SHELL=/bin/bash
ENV PIP_BREAK_SYSTEM_PACKAGES=1 

USER $USERNAME
ENV HOME=/home/$USERNAME
WORKDIR ${HOME}

CMD ["/bin/bash"]

###########################################
#  Fields2Cover
###########################################
FROM base AS fields2cover

USER root
WORKDIR ${HOME}

# Add ubuntugis PPA and install dependencies
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
	software-properties-common \
	build-essential \
	ca-certificates \
	cmake \
	doxygen \
	g++ \
	git \
	libeigen3-dev \
	libgdal-dev \
	libpython3-dev \
	python3 \
	python3-pip \
	lcov \
	libgtest-dev \
	libtbb-dev \
	swig \
	libgeos-dev \
	gnuplot \
	libtinyxml2-dev \
	nlohmann-json3-dev && \
	rm -rf /var/lib/apt/lists/*

# Install gcovr
RUN python3 -m pip install gcovr

# Clone and build Fields2Cover
USER $USERNAME
RUN git clone https://github.com/Rick-v-E/Fields2Cover.git && \
	cd ${HOME}/Fields2Cover && \
	mkdir -p build && cd build && \
	cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TUTORIALS=OFF -DBUILD_TESTS=OFF -DUSE_ORTOOLS_RELEASE=ON -DBUILD_PYTHON=ON .. && \
	make -j$(nproc)

USER root
RUN cd ${HOME}/Fields2Cover/build && make install

USER $USERNAME

CMD ["/bin/bash"]

###########################################
#  Develop image
###########################################
FROM fields2cover AS dev

USER root
WORKDIR ${HOME}

ENV DEBIAN_FRONTEND=noninteractive

# Install Latex
RUN apt-get update && apt-get install -y --no-install-recommends \
	texlive \
	dvipng \
	texlive-latex-extra \
	texlive-fonts-recommended \
	cm-super \
	&& rm -rf /var/lib/apt/lists/*

# Install gnome-terminal
RUN apt-get update && apt-get install -y --no-install-recommends \
	gnome-terminal \
	dbus-x11 \
	libcanberra-gtk-module \
	libcanberra-gtk3-module \
	&& rm -rf /var/lib/apt/lists/*

ENV NO_AT_BRIDGE=1

USER $USERNAME

CMD ["/bin/bash"]
