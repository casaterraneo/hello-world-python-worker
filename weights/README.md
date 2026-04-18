# Gestione pesi modelli

Questa cartella contiene gli script per esportare e caricare i pesi su Cloudflare KV.

## Procedura completa

### 1. Esporta i pesi (serve torch)

```powershell
& "C:\AS\autoresearch-my\.venv\Scripts\python.exe" weights\export_tris.py
& "C:\AS\autoresearch-my\.venv\Scripts\python.exe" weights\export_briscola.py
```

Genera nella stessa cartella:
- `tris_weights.bin` + `tris_hash.txt`
- `briscola_weights.bin` + `briscola_hash.txt`

### 2. Carica su KV

Usa la richiesta POST in `requests.http` (`### Upload tris weights` / `### Upload briscola weights`), oppure con PowerShell:

```powershell
$headers = @{
    "Content-Type"      = "application/octet-stream"
    "X-Internal-Secret" = "test-secret-1234"
    "X-Model-Name"      = "tris"          # oppure "briscola"
    "X-Model-Hash"      = "80e1805"       # hash dal file *_hash.txt
}
$bytes = [System.IO.File]::ReadAllBytes("$PWD\weights\tris_weights.bin")
Invoke-WebRequest -Uri "https://hello-python.casa-terraneo.workers.dev/admin/upload-weights" -Method POST -Headers $headers -Body $bytes
```

I `.bin` e `*_hash.txt` sono ignorati da git.

---

## Dove sono i modelli

| Modello  | File                                                                          |
|----------|-------------------------------------------------------------------------------|
| tris     | `C:\AS\autoresearch-my\tris\model.pt`                                         |
| briscola | `C:\AS\autoresearch-my\briscola2\best_models\model_0.8640_0f4a93c.pt`         |

Se aggiorni il modello briscola, cambia `MODEL_PATH` in `export_briscola.py`.
