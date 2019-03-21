FROM python:3.6

RUN pip install https://github.com/Cytomine-ULiege/LandmarkTools/archive/v0.0.4.zip
# --------------------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git
RUN cd /Cytomine-python-client && git checkout tags/v2.2.0 && pip install .
RUN rm -r /Cytomine-python-client

# --------------------------------------------------------------------------------------------
# Install Neubias-W5-Utilities (annotation exporter, compute metrics, helpers,...)
RUN git clone https://github.com/Neubias-WG5/neubiaswg5-utilities.git
RUN cd /neubiaswg5-utilities/ && git checkout tags/v0.5.3c && pip install .

# Metric for PixCla is pure python so don't need java, nor binaries
# RUN apt-get update && apt-get install openjdk-8-jdk -y && apt-get cleandock
# RUN chmod +x /neubiaswg5-utilities/bin/*
# RUN cp /neubiaswg5-utilities/bin/* /usr/bin/

RUN rm -r /neubiaswg5-utilities


RUN pip install scikit-learn imageio scipy sumproduct

ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py


ENTRYPOINT ["python", "/app/run.py"]
