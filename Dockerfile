# # Use AWS Lambda Python base image
# FROM public.ecr.aws/lambda/python:3.10

# WORKDIR /app

# # Install required system packages
# RUN yum update -y && yum install -y \
#     gcc \
#     python3 \
#     python3-devel \
#     libffi-devel \
#     make \
#     && yum clean all

# # Install pip
# RUN curl -O https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm -f get-pip.py


# # Copy dependencies first for caching
# COPY app/requirements.txt ./
# RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# # Copy application files into the Lambda task root
# COPY app/ "${LAMBDA_TASK_ROOT}/"

# # Set the Lambda function handler correctly
# CMD ["app.lambda_function.lambda_handler"]

FROM python:3.9
COPY app/requirements.txt /app/requirements.txt
WORKDIR /app

COPY app/ /app/

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]