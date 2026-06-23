# Deploy on AWS Free Tier

This guide walks through deploying the Company Knowledge Assistant (main app) and the Law Crawler on a fresh AWS free tier account. Everything described here stays within the AWS Free Tier limits as of 2026.

---

## Free Tier Budget

| Resource | Free Tier Limit | Our Usage | Cost |
|----------|----------------|-----------|------|
| EC2 t3.micro | 750 hrs/month (12 months) | 1 instance 24/7 = 744 hrs | $0 |
| RDS db.t3.micro PostgreSQL | 750 hrs/month (12 months) | 1 instance 24/7 = 744 hrs | $0 |
| EBS gp2 storage | 30 GB (EC2) + 20 GB (RDS) | ~15 GB total | $0 |
| Data transfer | 100 GB/month out | Minimal | $0 |

**No Redis/ElastiCache** — the app has MemoryRateLimiter, MemorySessionStore, and NoneCacheAdapter fallbacks. Redis is optional and skipped here.

---

## 1. Launch Infrastructure

### 1a. Create an RDS PostgreSQL Instance

Via AWS Console → RDS → Create database:

- **Engine**: PostgreSQL
- **Version**: 16.x or 17.x
- **Template**: Free tier
- **DB instance class**: `db.t3.micro`
- **Storage**: 20 GB gp2
- **DB instance identifier**: `cka-db`
- **Master username**: `postgres`
- **Master password**: generate a strong one (save it)
- **Connectivity**: Public access = **No** (EC2 connects via same VPC)
- **DB name**: `postgres` (leave default)
- **Auto-backup**: enabled (free up to 20 GB)

Note the **endpoint** (e.g., `cka-db.xxxxx.us-east-1.rds.amazonaws.com`).

### 1b. Launch an EC2 Instance

Via AWS Console → EC2 → Launch instance:

- **Name**: `cka-app`
- **AMI**: Ubuntu 24.04 LTS (free tier eligible)
- **Architecture**: 64-bit (x86)
- **Instance type**: `t3.micro`
- **Key pair**: Create or use an existing `.pem` key
- **Network**: Use the **same VPC** as the RDS instance
- **Security group** — create `cka-sg` with:

  | Type | Protocol | Port | Source |
  |------|----------|------|--------|
  | SSH | TCP | 22 | Your IP |
  | HTTP | TCP | 80 | 0.0.0.0/0 |
  | Custom TCP | TCP | 8000 | 0.0.0.0/0 |
  | Custom TCP | TCP | 8101-8103 | 0.0.0.0/0 |

- **Storage**: 20 GB gp2 (free tier includes 30 GB)
- **Advanced**: IAM instance profile — skip (no AWS SDK calls needed)

---

## 2. Configure EC2 Instance

### 2a. SSH In

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### 2b. Install Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git curl
sudo systemctl enable docker && sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

Log out and back in for the group change to take effect:

```bash
exit
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### 2c. Configure RDS Security Group

Go to the RDS security group and add an **inbound rule**:

- **Type**: PostgreSQL
- **Protocol**: TCP
- **Port**: 5432
- **Source**: The EC2 security group (`cka-sg`) — lookup its ID

This allows EC2 to reach RDS without exposing the database publicly.

---

## 3. Deploy the Main App

### 3a. Clone the Repository

```bash
git clone <your-repo-url> /home/ubuntu/cka
cd /home/ubuntu/cka
```

### 3b. Create `.env`

```bash
cat > .env << 'EOF'
DATABASE_URL=postgresql+asyncpg://postgres:<YOUR_PASSWORD>@cka-db.xxxxx.us-east-1.rds.amazonaws.com:5432/postgres
OPENAI_API_KEY=sk-...
JWT_SECRET=<generate-a-strong-random-string>
ADMIN_PASSWORD=<pick-a-strong-password>

# Disable Redis — fallback adapters are used
REDIS_URL=

# Disable MCP and A2A remote agents (in-process is fine)
MCP_ENABLED=false
A2A_LEGAL_RESEARCH_URL=
A2A_CITATION_CHECKER_URL=
A2A_RESPONSE_SYNTHESIZER_URL=
EOF
```

### 3c. Build and Run

```bash
docker compose build app
docker compose up -d postgres app
```

The app will fail on first start because the database is empty (no pgvector extension schema). This is expected.

### 3d. Run Database Migrations

```bash
# Install psql client
sudo apt install -y postgresql-client

# Apply migrations via Docker (if alembic is configured)
docker compose exec app alembic upgrade head

# OR manually create the pgvector extension
PGPASSWORD=<YOUR_PASSWORD> psql -h cka-db.xxxxx.us-east-1.rds.amazonaws.com -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3e. Verify It Works

```bash
curl http://localhost:8000/health
# → {"status":"ok"}

curl http://localhost:8000/
# → HTML page
```

---

## 4. Run the Law Crawler

The law crawler is a batch pipeline — it does not run as a service. Run it on demand on the same EC2 instance.

### 4a. Install Crawler Dependencies

```bash
cd /home/ubuntu/cka/law-crawler
pip install -r requirements.txt
```

### 4b. Run the Pipeline

```bash
# Full pipeline (Bronze → Silver → Gold)
python -m src.pipeline

# Resume from Silver layer (faster, for re-runs)
python -m src.pipeline --from silver

# Run only Gold layer
python -m src.pipeline --stage gold
```

Output lands in `law-crawler/data/gold/*.parquet`.

### 4c. Ingest Gold Data into the App

```bash
cd /home/ubuntu/cka
docker compose exec app python -c "
from app.factory import create_document_loader, create_vector_store, create_embeddings
from app.core.ingest_service import run_ingest
import asyncio

async def ingest():
    loader = create_document_loader()
    embeddings = create_embeddings()
    store = create_vector_store(embeddings)
    await run_ingest(loader, store, 'law-crawler/data/gold')
    print('Ingestion complete')

asyncio.run(ingest())
"
```

---

## 5. Test the Full Stack

```bash
# Get a JWT token
curl -X POST http://<EC2_PUBLIC_IP>:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"<ADMIN_PASSWORD>"}'

# Ask a question
TOKEN="eyJ..."
curl -X POST http://<EC2_PUBLIC_IP>:8000/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the labor laws in Vietnam?"}'
```

---

## 6. A2A Agents (Optional)

If you want to run agents as separate A2A servers:

```bash
docker compose up -d legal-research-agent citation-checker-agent response-synthesizer-agent
```

These use the same `Dockerfile.a2a` image. Set the `A2A_*_URL` env vars pointing to the EC2 instance's public IP. Note: t3.micro has 1 GB RAM — running all 3 A2A agents alongside the main app may cause memory pressure. Monitor with `docker stats`.

---

## 7. Automate the Law Crawler (Optional)

Schedule weekly data refreshes via cron on the EC2 instance:

```bash
crontab -e
```

Add:

```
# Every Sunday at 3 AM — run law crawler and ingest
0 3 * * 0 cd /home/ubuntu/cka/law-crawler && python -m src.pipeline --from silver && cd /home/ubuntu/cka && docker compose exec -T app python -c "import asyncio; from app.factory import create_document_loader, create_vector_store, create_embeddings; from app.core.ingest_service import run_ingest; asyncio.run(run_ingest(create_document_loader(), create_vector_store(create_embeddings()), 'law-crawler/data/gold'))"
```

This matches the GitHub Actions workflow (`law-crawler.yml`) but runs on EC2 instead of CI.

---

## 8. Staying Within Free Tier

### Watch These

| Gotcha | Why | How to Avoid |
|--------|-----|-------------|
| **EBS on stopped EC2** | EBS costs $ even when EC2 is stopped | Terminate, don't stop, when not in use |
| **Data transfer out** | First 100 GB/month free, then $0.09/GB | Keep API usage low; serve static files via CloudFront |
| **Elastic IP idle** | ~$3.60/month if unattached | Use the EC2's public IP (changes on stop/start) |
| **NAT Gateway** | ~$32/month if created | Don't create one; use a public subnet |

### Set a Billing Alarm

```bash
# Via AWS Console → CloudWatch → Alarms → Billing
# Create alarm: Total Estimated Charge > $0
# This emails you the moment any bill accrues
```

### Monthly Limit Check

- EC2: 744 hrs used out of 750 (99.2%) — OK
- RDS: 744 hrs used out of 750 (99.2%) — OK
- EBS: ~15 GB out of 50 GB combined — OK

---

## 9. Optional: All-in-One (PostgreSQL in Docker)

If you want to avoid RDS altogether and run everything on one EC2:

```bash
# Stop RDS, update .env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/postgres

# Run both services
docker compose up -d postgres app
```

This uses the Docker Compose PostgreSQL instead of RDS. **Memory warning**: t3.micro has 1 GB RAM. PostgreSQL alone uses ~200-300 MB, leaving ~700 MB for the app. This is fine for low traffic but monitor with `free -h` and `docker stats`.

---

## 10. Architecture Diagram (Free Tier)

```mermaid
graph TB
    subgraph AWS["AWS Cloud (Free Tier)"]
        subgraph VPC["Default VPC"]
            subgraph PublicSubnet["Public Subnet"]
                EC2["EC2 t3.micro<br/>cka-app<br/>Docker: app + A2A agents"]
            end
            subgraph PrivateSubnet["Private Subnet"]
                RDS["RDS db.t3.micro<br/>PostgreSQL + pgvector"]
            end
            SG["Security Group<br/>cka-sg"]
        end
    end

    Client["You / API Consumers"] -->|HTTP port 8000| EC2
    EC2 -->|asyncpg port 5432| RDS
    EC2 -->|SSH port 22| Admin["Admin (your IP only)"]
    SG -.-> EC2
    SG -.-> RDS
```

---

## Cost Summary

If you follow this guide exactly and stay within free tier limits for 12 months:

| Service | Monthly Cost |
|---------|-------------|
| EC2 t3.micro | $0 |
| RDS db.t3.micro | $0 |
| EBS (ECI + RDS) | $0 |
| Data transfer (~2 GB) | $0 |
| **Total** | **$0** |

After the 12-month free tier expires, the same setup costs ~$25-30/month (EC2 ~$8, RDS ~$17, EBS ~$1, data ~$1).
