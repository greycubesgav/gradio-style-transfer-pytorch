FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Tell Debian based distro to not ask for any input during installs
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHON_ENV='st-pytorch'

# Create a local user
ENV USER=pythonapp
RUN useradd --create-home --home-dir "/home/${USER}/" --shell /bin/bash --uid 1000 --user-group -G users "${USER}"

# Switch this user
USER ${USER}

# Switch to this directory
WORKDIR /home/${USER}

# Set the HOME environment variable
ENV HOME=/home/${USER}

# Copy the source content to the container
COPY --chown=${USER}:users style_transfer /${HOME}/style_transfer

# Install the python requirements in the base OS (we're not using conda here, just the root python install provided by pytorch image)
RUN /bin/bash --login -c "pip install -r \"${HOME}/style_transfer/requirements.txt\""

# Expose a port for the inprogress webserver
EXPOSE 8080

# Run bash
ENTRYPOINT [ "bash" ]