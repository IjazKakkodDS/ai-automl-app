FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip

# pywin32 is a Windows-only package present because requirements.txt was
# generated on Windows. Filter it before installing on the Linux container.
RUN grep -v "^pywin32" requirements.txt > /tmp/req_linux.txt \
    && pip install -r /tmp/req_linux.txt

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
