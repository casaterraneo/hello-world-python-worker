import json
import base64
import struct
from urllib.parse import urlparse
from workers import WorkerEntrypoint, Response

HARDCODED_SECRET = "test-secret-1234"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_resp(data, status=200):
    return Response(
        json.dumps(data, separators=(",", ":")),
        status=status,
        headers={"Content-Type": "application/json"},
    )

def _error(msg, status=400):
    return _json_resp({"detail": msg}, status=status)

# ---------------------------------------------------------------------------
# Tris — inferenza pure Python (pesi estratti da model.pt via export_weights.py)
# Nessuna dipendenza da numpy: usa solo struct (stdlib)
# ---------------------------------------------------------------------------

_tris_weights = None

# Shapes: w0(256x9) b0(256) w2(256x256) b2(256) w4(256x256) b4(256) w6(9x256) b6(9)
_W_SHAPES = [(256, 9), (256,), (256, 256), (256,), (256, 256), (256,), (9, 256), (9,)]
_W_NAMES  = ["w0", "b0", "w2", "b2", "w4", "b4", "w6", "b6"]

def _get_tris_weights():
    global _tris_weights
    if _tris_weights is None:
        from tris_weights import WEIGHTS_B64
        raw = base64.b64decode(WEIGHTS_B64)
        all_floats = struct.unpack(f"{len(raw)//4}f", raw)
        offset = 0
        weights = {}
        for name, shape in zip(_W_NAMES, _W_SHAPES):
            n = 1
            for s in shape:
                n *= s
            weights[name] = list(all_floats[offset:offset + n])
            offset += n
        _tris_weights = weights
    return _tris_weights

def _linear_relu(W, b, x, out_f, in_f):
    return [max(0.0, sum(W[i * in_f + j] * x[j] for j in range(in_f)) + b[i])
            for i in range(out_f)]

def _linear(W, b, x, out_f, in_f):
    return [sum(W[i * in_f + j] * x[j] for j in range(in_f)) + b[i]
            for i in range(out_f)]

def _tris_forward(board):
    w = _get_tris_weights()
    x = [float(v) for v in board]
    x = _linear_relu(w["w0"], w["b0"], x, 256, 9)
    x = _linear_relu(w["w2"], w["b2"], x, 256, 256)
    x = _linear_relu(w["w4"], w["b4"], x, 256, 256)
    q = _linear(w["w6"], w["b6"], x, 9, 256)
    legal = [i for i, v in enumerate(board) if v == 0]
    if not legal:
        raise ValueError("Nessuna mossa legale disponibile")
    return max(legal, key=lambda i: q[i])

# ---------------------------------------------------------------------------
# Briscola — encode osservazione + inferenza pure Python
# ---------------------------------------------------------------------------

_SUITS       = ["bastoni", "coppe", "denari", "spade"]
_POINTS      = {1: 11, 3: 10, 10: 4, 9: 3, 8: 2}
_BRISCOLA_W  = None
_B_NAMES     = ["w0", "b0", "w2", "b2", "w4", "b4"]
_B_SHAPES    = [(256, 31), (256,), (256, 256), (256,), (3, 256), (3,)]

def _get_briscola_weights():
    global _BRISCOLA_W
    if _BRISCOLA_W is None:
        from briscola_weights import WEIGHTS_B64
        raw = base64.b64decode(WEIGHTS_B64)
        all_floats = struct.unpack(f"{len(raw)//4}f", raw)
        offset = 0
        weights = {}
        for name, shape in zip(_B_NAMES, _B_SHAPES):
            n = 1
            for s in shape:
                n *= s
            weights[name] = list(all_floats[offset:offset + n])
            offset += n
        _BRISCOLA_W = weights
    return _BRISCOLA_W

def _encode_card(card):
    """card: {"suit": str, "rank": int} oppure None -> [0]*5"""
    if card is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    suit  = card.get("suit", "").lower()
    rank  = int(card.get("rank", 0))
    if suit not in _SUITS or not 1 <= rank <= 10:
        return None  # segnala errore
    pts = _POINTS.get(rank, 0)
    return [
        rank / 10.0,
        pts  / 11.0,
        float(suit == "bastoni"),
        float(suit == "coppe"),
        float(suit == "denari"),
        # spade è implicita (tutti e tre i flag a 0)
    ]

def _briscola_encode(body):
    """Restituisce (obs: list[float], error: str|None)"""
    hand     = body.get("hand")
    table    = body.get("table")       # può essere null
    briscola = body.get("briscola")
    deck_left  = body.get("deck_left")
    my_score   = body.get("my_score")
    opp_score  = body.get("opp_score")

    if not isinstance(hand, list) or not 1 <= len(hand) <= 3:
        return None, "hand deve essere una lista di 1-3 carte"
    if briscola is None:
        return None, "briscola e' obbligatorio"
    if deck_left is None or my_score is None or opp_score is None:
        return None, "deck_left, my_score e opp_score sono obbligatori"

    obs = []
    # 3 carte in mano (pad con zeros se meno di 3)
    for i in range(3):
        c = hand[i] if i < len(hand) else None
        enc = _encode_card(c)
        if enc is None:
            return None, f"carta in hand[{i}] non valida"
        obs.extend(enc)

    # carta sul tavolo
    enc = _encode_card(table)
    if enc is None:
        return None, "carta in table non valida"
    obs.extend(enc)

    # briscola
    enc = _encode_card(briscola)
    if enc is None:
        return None, "carta briscola non valida"
    obs.extend(enc)

    # scalari normalizzati
    obs.append(float(deck_left) / 36.0)
    obs.append(float(my_score)  / 120.0)
    obs.append(float(opp_score) / 120.0)

    # batte_tavolo: 3 flag, uno per carta in mano
    _RANK_ORDER = [1, 3, 10, 9, 8, 7, 6, 5, 4, 2]
    briscola_suit = briscola.get("suit", "").lower()
    table_suit  = table.get("suit", "").lower()  if table else None
    table_rank  = int(table.get("rank", 0))      if table else None
    for i in range(3):
        if i >= len(hand) or table is None or table_suit not in _SUITS or not 1 <= (table_rank or 0) <= 10:
            obs.append(0.0)
            continue
        c = hand[i]
        c_suit = c.get("suit", "").lower()
        c_rank = int(c.get("rank", 0))
        p_bris = c_suit == briscola_suit
        t_bris = table_suit == briscola_suit
        if p_bris and not t_bris:
            wins = True
        elif t_bris and not p_bris:
            wins = False
        elif c_suit == table_suit:
            wins = _RANK_ORDER.index(c_rank) < _RANK_ORDER.index(table_rank)
        else:
            wins = False
        obs.append(1.0 if wins else 0.0)

    return obs, None

def _briscola_forward(obs, n_cards=3):
    w = _get_briscola_weights()
    x = obs[:]
    x = _linear_relu(w["w0"], w["b0"], x, 256, 31)
    x = _linear_relu(w["w2"], w["b2"], x, 256, 256)
    q = _linear(w["w4"], w["b4"], x, 3, 256)
    legal = list(range(n_cards))
    return max(legal, key=lambda i: q[i])

# ---------------------------------------------------------------------------
# Worker entrypoint — routing nativo senza FastAPI
# ---------------------------------------------------------------------------

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        path = urlparse(str(request.url)).path
        method = str(request.method)

        # GET /
        if method == "GET" and path == "/":
            await self.env.KV_BINDING.put("bar", "baz")
            bar = await self.env.KV_BINDING.get("bar")
            message = f"Hello world TEST! Version: {self.env.APP_VERSION} {bar}"
            return _json_resp({"message": message})

        # GET /test-depends-async
        if method == "GET" and path == "/test-depends-async":
            key = request.headers.get("x-internal-secret")
            if key != HARDCODED_SECRET:
                return _error("Unauthorized", status=401)
            return _json_resp({"auth": "ok", "method": "Depends asincrona"})

        # POST /tris/move
        if method == "POST" and path == "/tris/move":
            try:
                body = json.loads(await request.text())
            except Exception:
                return _error("JSON non valido")
            board = body.get("board") if isinstance(body, dict) else None
            if not isinstance(board, list) or len(board) != 9:
                return _error("board deve avere esattamente 9 valori")
            for v in board:
                if v not in (0, 1, -1, 0.0, 1.0, -1.0):
                    return _error("I valori del board devono essere 0, 1 o -1")
            move = _tris_forward(board)
            return _json_resp({"move": move})

        # POST /briscola/encode
        if method == "POST" and path == "/briscola/encode":
            try:
                body = json.loads(await request.text())
            except Exception:
                return _error("JSON non valido")
            obs, err = _briscola_encode(body)
            if err:
                return _error(err)
            return _json_resp({"obs": obs})

        # POST /briscola/move
        if method == "POST" and path == "/briscola/move":
            try:
                body = json.loads(await request.text())
            except Exception:
                return _error("JSON non valido")
            obs, err = _briscola_encode(body)
            if err:
                return _error(err)
            n_cards = int(body.get("n_cards", 3))
            if not 1 <= n_cards <= 3:
                return _error("n_cards deve essere 1, 2 o 3")
            move = _briscola_forward(obs, n_cards)
            return _json_resp({"move": move})

        return _error("Not found", status=404)
