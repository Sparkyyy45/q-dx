# CardioQ Clinical Workstation: Production Deployment Guide

This guide provides step-by-step instructions for deploying CardioQ across all environments: from an **instant 30-second public demo link** for judges to **free cloud hosting**, **Docker containers**, and **enterprise Linux servers**.

---

## ⚡ Quick Decision Guide: Which Method Should You Use?

| Scenario | Recommended Method | Time to Deploy | Cost |
| :--- | :--- | :--- | :--- |
| **Instant Live Demo for Judges / Reviewers** | **Method 1: ngrok / Cloudflare Tunnel** | **30 seconds** | Free |
| **24/7 Public Cloud URL (GitHub Integration)** | **Method 2: Render.com / Railway** | **3 minutes** | Free tier |
| **AI / Machine Learning Showcase** | **Method 3: Hugging Face Spaces** | **5 minutes** | Free |
| **Self-Hosted / Cloud VM / On-Premises** | **Method 4: Docker & Docker Compose** | **2 minutes** | Own Server |
| **Hospital Network / Production Linux Server** | **Method 5: Ubuntu VM + Systemd + Nginx** | **10 minutes** | Cloud VPS |

---

## Method 1: Instant Live Public Link (30 Seconds)
*Best for giving judges or evaluators a live, secure HTTPS link to test on their own phones/laptops right now.*

Your local CardioQ server is already running on `http://127.0.0.1:8080`. You can generate an encrypted public URL instantly without uploading files:

### Option A: Using Cloudflare Tunnel (No account or login required!)
Run in your terminal:
```bash
npx untun@latest tunnel http://localhost:8080
```
*Or download `cloudflared` from Cloudflare and run:*
```bash
cloudflared tunnel --url http://localhost:8080
```
You will get an instant public URL like `https://random-words.trycloudflare.com` that anyone can open immediately.

### Option B: Using ngrok
1. If you have `ngrok` installed:
   ```bash
   ngrok http 8080
   ```
2. Copy the `https://xxxx.ngrok-free.app` URL and share it.

---

## Method 2: Free 24/7 Cloud Hosting on Render.com
*Best for having a permanent public URL linked to your GitHub repository.*

The repository now includes `render.yaml`, `Procfile`, and `requirements.txt`.

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "feat: production deployment ready"
   git push origin main
   ```
2. Go to [Render.com](https://render.com) and click **"New +" -> "Web Service"**.
3. Connect your GitHub repository.
4. Render will automatically detect the settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `python app.py --host 0.0.0.0 --port $PORT`
5. Click **Deploy Web Service**.
   Your app will be live at `https://cardioq-xxxx.onrender.com`.

---

## Method 3: Hugging Face Spaces (Free AI Hosting)
*Perfect for Health AI & Machine Learning hackathon submissions.*

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set Space SDK to **Docker** (Blank).
3. Clone your Space repo locally and push the CardioQ repository contents (including `Dockerfile` and `requirements.txt`).
4. Hugging Face will automatically build and launch the container with a permanent public `.hf.space` URL and an embedded preview.

---

## Method 4: Production Docker Deployment
*For containerized deployment on any cloud server, local machine, or Kubernetes cluster.*

The repository contains a production-ready multi-stage `Dockerfile` and `docker-compose.yml`.

### Single Container Run:
```bash
# 1. Build the Docker image
docker build -t cardioq:latest .

# 2. Run the container on port 8080
docker run -d --name cardioq -p 8080:8080 --restart unless-stopped cardioq:latest
```

### Multi-Container via Docker Compose (with persistent database):
```bash
docker compose up -d
```
- Web dashboard accessible at: `http://localhost:8080`
- SQLite history and uploaded datasets are persisted in the local `./artifacts` folder.
- Stop with: `docker compose down`

---

## Method 5: Enterprise Linux Server (Ubuntu / AWS EC2 / DigitalOcean)
*For enterprise hospital network or permanent cloud VPS deployment.*

### 1. Clone & Setup Python Virtual Environment:
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git nginx

git clone <your-repo-url> /opt/cardioq
cd /opt/cardioq

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Systemd Background Service:
Create `/etc/systemd/system/cardioq.service`:
```ini
[Unit]
Description=CardioQ Clinical Workstation Engine
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/cardioq
ExecStart=/opt/cardioq/venv/bin/python app.py --host 127.0.0.1 --port 8080
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=CARDIOQ_HOST=127.0.0.1
Environment=CARDIOQ_PORT=8080

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable cardioq
sudo systemctl start cardioq
sudo systemctl status cardioq
```

### 3. Setup Nginx Reverse Proxy with SSL:
Create `/etc/nginx/sites-available/cardioq`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 25M;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable Nginx site and secure with free HTTPS:
```bash
sudo ln -s /etc/nginx/sites-available/cardioq /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 🔒 Verification & Health Check Endpoints

Once deployed, verify that the platform is healthy:
- **Web UI**: `GET /` -> Returns status 200 with CardioQ interface
- **Health Check**: `GET /health` -> `{"status": "healthy", "version": "2.0.0"}`
- **Model Inference**: `POST /api/predict` -> Returns calibrated probability & SHAP values
- **Quantum Bridge**: `POST /api/quantum/qiskit/execute` -> Returns NISQ simulation results
