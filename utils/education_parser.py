
import re, os, json, functools, requests
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# ─── degree-level rules ──────────────────────────────────────────────
LEVEL_SYNS = {
    4: ["phd","doctorate","doctoral","dphil","edd","doctor of"],
    3: ["master","msc","m.sc","m.s","mba","m.eng","m.tech"],
    2: ["bachelor","bsc","b.sc","bs ","ba ","b.tech","b.eng","be "],
    1: ["associate","aas","a.a.s","aa ","a.s.","associates"],
    0: ["high school","secondary","ged","h.s"]
}
LEVEL_CANON = {k:v[0].split()[0].title() for k,v in LEVEL_SYNS.items()}
LEVEL_RX    = {k:re.compile(rf"\b({'|'.join(v)})\b",re.I)
               for k,v in LEVEL_SYNS.items()}

def _rule_level(txt):
    for k in sorted(LEVEL_RX,reverse=True):
        if LEVEL_RX[k].search(txt): return k
    return None

FUZZ_TGT = ["Doctor","Master","Bachelor","Associate","High School"]
def _fuzzy_level(txt):
    best,score,_ = process.extractOne(txt,FUZZ_TGT,scorer=fuzz.WRatio)
    return FUZZ_TGT.index(best) if score>=88 else None

# ─── MiniLM embedding -- catches odd phrasings ───────────────────────
_embed      = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_canon_emb  = _embed.encode([f"{LEVEL_CANON[k]} degree" for k in range(5)],
                            normalize_embeddings=True)
def _embed_level(txt):
    v = _embed.encode([txt],normalize_embeddings=True)
    sims = util.cos_sim(v,_canon_emb)[0]
    best = int(sims.argmax())
    return best if sims[best]>0.5 else None

# ─── canonical majors -- small synonym map (extend as needed) ────────
MAJOR_MAP = {
    "Computer Science": ["computer science","comp sci","cse","cs","software eng"],
    "Information Tech" : ["information technology","info tech"," it "],
    "Business Admin"   : ["business administration","business studies","bba"],
    "Electrical Eng"   : ["electrical eng","ee","electrical engineering"],
    "Mechanical Eng"   : ["mechanical eng"],
    "Accounting"       : ["accounting","accountancy"],
    "Data Science"     : ["data science"],
    "Public Health"    : ["public health","mph"]
}
def _canon_major(raw):
    if not raw: return None
    low=raw.lower()
    for canon,toks in MAJOR_MAP.items():
        if any(t in low for t in toks): return canon
    return None

# ─── Mistral fallback (only for degree level) ────────────────────────
load_dotenv()
_KEY=os.getenv("MISTRAL_API_KEY")
_URL="https://api.mistral.ai/v1/chat/completions"
_FUNC={"name":"parse_degree","parameters":{"type":"object","properties":{
       "level":{"type":"string"},"major":{"type":"string"},
       "is_degree":{"type":"boolean"}}}}

@functools.lru_cache(maxsize=5000)
def _mistral(prompt:str):
    if not (_KEY and prompt): return None
    body=dict(model="mistral-large-latest",temperature=0,
              messages=[{"role":"system","content":"Extract highest academic degree."},
                        {"role":"user","content":prompt}],
              tools=[{"type":"function","function":_FUNC}],
              tool_choice={"name":"parse_degree"})
    r=requests.post(_URL,json=body,headers={"Authorization":f"Bearer {_KEY}"},timeout=20)
    if r.status_code!=200: return None
    args=json.loads(r.json()["choices"][0]["message"]["tool_calls"][0]
                    ["function_call"]["arguments"])
    lvl_txt=(args.get("level") or "").lower()
    lvl=next((k for k,v in LEVEL_SYNS.items() if any(t in lvl_txt for t in v)),None)
    return lvl

# ─── public function: parse_degree(raw_str) ──────────────────────────
def parse_degree(raw:str):
    if not raw or not raw.strip():
        return {"level_code":None,"level_name":None,"major":None,"is_degree":False}
    txt=re.sub(r"[^a-z0-9: ]+"," ",raw.lower())
    lvl=_rule_level(txt) or _fuzzy_level(txt) or _embed_level(txt)
    # extract major phrase (if ":" present or " in <major>")
    major_raw=None
    if ":" in raw: major_raw=raw.split(":",1)[1].strip()
    else:
        m=re.search(r"\bin\s+(.+)",raw,re.I)
        if m: major_raw=m.group(1).strip()
    major=_canon_major(major_raw or "")
    if lvl is None: lvl=_mistral(raw)
    return {"level_code":lvl,
            "level_name":LEVEL_CANON.get(lvl),
            "major":major,
            "is_degree":bool(lvl is not None)}
