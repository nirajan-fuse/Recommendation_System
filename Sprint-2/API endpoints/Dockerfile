FROM continuumio/miniconda3

WORKDIR /app

# Update Conda
RUN conda update -n base -c defaults conda

# create the conda environment
COPY environment.yml .

# Update Conda and create the Conda environment
RUN conda env create -f environment.yml --platform win-64 

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port that app runs on
EXPOSE 8000

# # Activate the environment and run the FastAPI application
CMD ["conda", "run", "--no-capture-output", "-n", "env", "uvicorn", "main:app", "--host", "0.0.0.0", "--reload", "--port", "8000"]