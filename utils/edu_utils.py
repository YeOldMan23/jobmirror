


"""
This is the main utils file for extracting resume education information.
for brzone to silver transformation.
"""









# utils/edu_utils.py  – regex + rapid-fuzz only
import re, unicodedata
from rapidfuzz import process, fuzz

# ─── helpers ────────────────────────────────────────────
def _norm(t):
    if not t: return ""
    t = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode()
    return t.lower().strip()

# ─── DEGREE-LEVEL synonyms ──────────────────────────────
LEVEL_SYNONYMS = {
    # Doctorate
    "doctor": "doctorate", "doctorate":"doctorate", "phd":"doctorate",
    "ph.d":"doctorate", "dba":"doctorate", "phd of science":"doctorate",
    "juris doctor":"doctorate", "jd":"doctorate",
    # Master
    "master": "master", "masters": "master", "master degree":"master",
    "msc":"master", "m.sc":"master", "ms ":"master", "m.s":"master",
    "meng":"master", "m.eng":"master", "mba":"master", "m.b.a":"master",
    "macc":"master", "master of accountancy":"master",
    # Bachelor
    "bachelor":"bachelor", "bachelors":"bachelor", "bachelor's":"bachelor",
    "bs ":"bachelor", "b.s":"bachelor", "bsc":"bachelor", "b.sc":"bachelor",
    "ba ":"bachelor", "b.a":"bachelor", "beng":"bachelor", "b.eng":"bachelor",
    "beng (hons)":"bachelor", "b.tech":"bachelor", "btech":"bachelor",
    "be ":"bachelor", "b.e":"bachelor", "bba":"bachelor", "b.s.":"bachelor",
    # Associate
    "associate":"associate", "associates":"associate", "associate degree":"associate",
    "aa ":"associate", "a.a":"associate", "aas":"associate", "a.a.s":"associate",
    "as ":"associate", "a.s":"associate",
    # Diploma / certificate / course / secondary
    "post graduate diploma":"diploma", "pg diploma":"diploma", "diploma":"diploma",
    "technical diploma":"diploma", "certificate":"certificate",
    "certification":"certificate", "course on":"certificate",
    "license":"certificate",
    "high school":"secondary", "secondary school":"secondary",
    "ged":"secondary", "some college":"secondary"
}
_LEVEL_KEYS = list(LEVEL_SYNONYMS.keys())

# ─── MAJOR list  (≈120) ─────────────────────────────────
MAJOR_LIST = list({
    # engineering / stem
    "computer science","computer engineering","software engineering",
    "information technology","information systems","data science",
    "data analytics","business analytics","statistics","mathematics",
    "applied mathematics","electronics engineering",
    "electronics and communication engineering",
    "electrical engineering","electrical and computer engineering",
    "electrical electronics and communications engineering",
    "electronics & telecommunication","telecommunications engineering",
    "telecommunications","civil engineering","chemical engineering",
    "mechanical engineering","mechatronics","biomedical engineering",
    "biomedical diagnostics","biological sciences","biology",
    "healthcare administration","public health","health administration",
    "industrial engineering","industrial microbiology",
    "resource economics","applied economics",
    "environmental studies","environmental science",
    # business / social / other
    "business administration","business studies","finance",
    "accounting","commerce","management information systems",
    "management","marketing","economics","political science",
    "sociology","psychology","liberal arts","criminal justice",
    "law","human resource management","project management",
    "hospitality and tourism management","education",
    "electrical occupations","cosmetology","nursing","media arts"
})

GPA_RE = re.compile(r"(\\d+\\.\\d{1,2}|\\d)(?=\\s*/?\\s*10?\\.?0?)")

# ─── extraction helpers ─────────────────────────────────
def level_from_text(txt:str):
    low = _norm(txt)
    for k,v in LEVEL_SYNONYMS.items():
        if k in low: return v, "regex"
    m,score,_ = process.extractOne(low, _LEVEL_KEYS, scorer=fuzz.partial_ratio)
    return (LEVEL_SYNONYMS[m],"fuzzy") if score>=88 else (None,None)

def major_from_text(txt:str):
    low = _norm(txt)
    for m in MAJOR_LIST:
        if m in low: return m, "regex"
    m2,score,_ = process.extractOne(low, MAJOR_LIST, scorer=fuzz.partial_ratio)
    return (m2,"fuzzy") if score>=85 else (None,None)

def gpa_from_text(txt:str):
    m = GPA_RE.search(txt or "")
    return float(m.group()) if m else None
