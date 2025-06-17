


"""
This is the main utils file for extracting resume education information.
for brzone to silver transformation.
"""







from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql import Row

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

education_levels = [
    Row(group_name="High School", 
        group_scale=1, 
        group_description="Completion of high school or equivalent",
        group_references=[
            "high school", "ged", "secondary", "college", "diploma"
        ]),
    Row(group_name="Associate's Degree", 
        group_scale=2, 
        group_description="Completion of a two-year degree program or certificate",
        group_references=[
            "associate's", "a.a", "a.s", "a.a.s", "certificate"
        ]),
    Row(group_name="Bachelor's Degree", 
        group_scale=3, 
        group_description="Completion of an undergraduate degree program",
        group_references=[
            "bachelor", "bachelor's degree", "bachelor's in", "undergraduate degree", "ba", "bs", "bsc", "bba", "beng"
        ]),
    Row(group_name="Master's Degree", 
        group_scale=4, 
        group_description="Completion of a postgraduate degree program",
        group_references=[
            "master", "master's degree", "master's in", "postgraduate degree", "ma", "ms", "msc", "mba", "m.eng"
        ]),
    Row(group_name="Doctorate", 
        group_scale=5, 
        group_description="Completion of a doctoral degree program",
        group_references=[
            "phd", "md", "jd", "doctor", "doctorate", "doctorate degree", "doctoral degree"
        ])
]

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


education_fields = [
    Row(group_name="Computer Science & IT", 
        group_references=[
            "computer science", "software engineering", "information technology", "data science",
            "artificial intelligence", "cybersecurity", "machine learning", "informatics",
            "information systems", "computer engineering", "ai", "it"
    ]),
    Row(group_name="Engineering (General)", 
        group_references=[
            "mechanical engineering", "electrical engineering", "civil engineering", "chemical engineering",
            "industrial engineering", "engineering", "systems engineering", "aerospace engineering",
            "mechatronics", "structural engineering"
    ]),
    Row(group_name="Mathematics & Statistics", 
        group_references=[
            "mathematics", "applied mathematics", "statistics", "mathematical sciences",
            "pure mathematics", "quantitative", "data analytics", "operations research", "computational mathematics"
    ]),
    Row(group_name="Business & Management", 
        group_references=[
            "business administration", "management", "operations management", "organizational leadership",
            "strategic management", "entrepreneurship", "general management", "business",
            "project management", "business management"
    ]),
    Row(group_name="Finance & Accounting", 
        group_references=[
            "finance", "accounting", "financial engineering", "banking", "audit", "taxation",
            "financial analysis", "investment", "corporate finance", "actuarial studies"
    ]),
    Row(group_name="Economics", 
        group_references=[
            "economics", "development economics", "applied economics", "econometrics", 
            "macroeconomics", "microeconomics", "international economics", "economic policy",
            "quantitative economics", "political economy"
    ]),
    Row(group_name="Marketing & Communications", 
        group_references=[
            "marketing", "public relations", "advertising", "digital marketing", "communications",
            "media studies", "brand management", "corporate communications", "content marketing", 
            "strategic communications"
    ]),
    Row(group_name="Humanities", 
        group_references=[
                "history", "philosophy", "literature", "cultural studies", "classical studies",
                "theology", "religious studies", "humanities", "ethics", "liberal arts"
    ]),
    Row(group_name="Social Sciences", 
        group_references=[
            "sociology", "political science", "psychology", "anthropology", "international relations",
            "social work", "gender studies", "criminology", "public policy", "social sciences"
    ]),
    Row(group_name="Education & Training", 
        group_references=[
            "education", "pedagogy", "curriculum and instruction", "educational leadership", "teacher training",
            "educational psychology", "early childhood education", "instructional design",
            "higher education", "adult education"
    ]),
    Row(group_name="Law & Legal Studies", 
        group_references=[
            "law", "llb", "jurisprudence", "legal studies", "international law", "corporate law",
            "human rights law", "civil law", "criminal law", "constitutional law"
    ]),
    Row(group_name="Medicine & Health Sciences", 
        group_references=[
            "medicine", "nursing", "dentistry", "public health", "pharmacy", "clinical medicine",
            "healthcare administration", "biomedical sciences", "physiotherapy", "occupational therapy"
    ]),
    Row(group_name="Biological Sciences", 
        group_references=[
            "biology", "biochemistry", "molecular biology", "zoology", "genetics", 
            "microbiology", "botany", "life sciences", "cell biology", "marine biology"
    ]),
    Row(group_name="Physical Sciences", 
        group_references=[
            "physics", "chemistry", "geology", "earth sciences", "astronomy", 
            "material science", "astrophysics", "theoretical physics", "nuclear physics", "physical sciences"
    ]),
    Row(group_name="Environmental & Agricultural Sciences", 
        group_references=[
            "environmental science", "agriculture", "forestry", "horticulture", "agroecology",
            "natural resource management", "soil science", "sustainable agriculture", "environmental studies", "wildlife biology"
    ]),
    Row(group_name="Architecture & Design", 
        group_references=[
            "architecture", "interior design", "urban planning", "landscape architecture",
            "industrial design", "environmental design", "urban design", "architectural engineering",
            "built environment", "spatial design"
    ]),
    Row(group_name="Arts & Creative Fields", 
        group_references=[
            "fine arts", "graphic design", "visual arts", "performing arts", "music",
            "art history", "photography", "fashion design", "theatre", "creative writing"
    ]),
    Row(group_name="Language & Linguistics", 
        group_references=[
            "linguistics", "translation", "modern languages", "applied linguistics", "literary studies",
            "english language", "foreign languages", "language studies", "philology", "interpreting"
    ]),
    Row(group_name="Interdisciplinary", 
        group_references=[
            "general studies", "interdisciplinary studies", "liberal studies", "science and technology studies",
            "cognitive science", "international studies", "innovation studies", "multidisciplinary studies",
            "global studies", "unknown"
    ]),
]

# -------------- Certifications -------------------
certification_categories = [
    Row(group_name="Cloud Certifications", group_references=[
        "aws", "aws certifications", "aws certifications", "azure", "microsoft azure",
        "azure devops", "azure app services", "azure storage", "gcp", "google cloud",
        "oracle cloud applications", "oracle cloud infrastructure", "snowflake", "alteryx",
    ]),
    Row(group_name="DevOps & Infrastructure", group_references=[
        "docker", "kubernetes", "jenkins", "terraform", "ansible", "rundeck",
        "cloudformation", "git", "maven", "grunt"
    ]),
    Row(group_name="Cybersecurity Certifications", group_references=[
        "cissp", "security+", "cysa+", "gicsp", "hipaa", "iso26262",
        "aspice", "sso", "saml", "openid connect"
    ]),
    Row(group_name="Data & AI Tools", group_references=[
        "mlflow", "databricks", "dbt", "sisense", "tableau", "power automate",
        "periscope", "bi tools", "ssis", "airflow", "boomi", "kafka", "spark",
        "hadoop", "hive", "pig", "cassandra", "mongodb"
    ]),
    Row(group_name="Accounting & Finance", group_references=[
        "cpa", "certified public accountant", "cpa license", "cpa (preferred)",
        "cma", "quickbooks certified", "floqast", "copas accounting procedure",
        "cpa (working towards certification)", "cpa (certified public accountant)"
    ]),
    Row(group_name="Business Analysis", group_references=[
        "cbap", "ccba", "certification of competency in business analysis (ccba)",
        "certified business analysis professional (cbap)", "six sigma",
        "design for six sigma", "product owner", "product owner certification",
        "capm", "pmp"
    ]),
    Row(group_name="Agile & Project Management", group_references=[
        "scrum master", "scrum master certification", "safe scrum master",
        "safe", "safe scrum master", "scrum", "pmp", "agile product owner",
        "product owner", "project management professional"
    ]),
    Row(group_name="ERP & Business Software", group_references=[
        "salesforce", "salesforce certification", "salesforce omni studio developer",
        "salesforce certified administrator", "salesforce platform developer i",
        "salesforce platform developer ii", "salesforce platform app builder",
        "sap r3", "sap s4 (fi)", "microsoft dynamics 365", "oracle cloud",
        "jde edwards", "netsuite", "sage intacct"
    ]),
    Row(group_name="Programming & Development Tools", group_references=[
        "visual studio", ".net framework", "html", "javascript", "jquery",
        "typescript", "c#", "entity framework", "webapi", "code composer studio",
        "canalyzer", "canoe", "c/c++", "python", "java", "kotlin",
        "swift", "go", "ruby", "php", "rust", "scala",
    ]),
    Row(group_name="Engineering & Technical", group_references=[
        "pe", "professional engineer license", "professional engineering (pe) license",
        "eit", "eit certification in an energy-related field",
        "labview", "xilinx fpga", "altera fpga", "ea", "ce (computer environment) linux+"
    ]),
    Row(group_name="Legal & Regulatory", group_references=[
        "notary", "hipaa", "dcaa", "dcma", "niccs work role id: pr-vam-001",
        "niccs work role id: an-exp-001", "certification or progress toward certification is highly preferred and encouraged"
    ]),
    Row(group_name="Driver's Licenses", group_references=[
        "valid california driver's license", "california driver's license", "commercial driver's license"
    ]),
    Row(group_name="System Administration", group_references=[
        "csa", "csa (servicenow certified system administrator)",
        "csa (certified system administrator)", "linux+", "windows server",
        "azure active directory", "active directory", "vmware",
        "vmware vcp", "vmware certified professional", "vmware vcp-dcv"
    ]),
]


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

def save_education_synonyms(spark: SparkSession, filepath:str, data:list[Row]) -> None:
    """
    Creates a parquet file with education level synonyms as well as the corresponding scale.
    """
    education = spark.createDataFrame(data)
    # Save as Parquet (to preserve schema and array types)
    education.write.mode('overwrite').parquet(filepath)
    return

def determine_edu_mapping(education_input: str | list[str], mapping: list, threshold: int=50, ) -> str | None:
    """
    Determine the mapping for a given input string for possible education values (level/field).
    Uses fuzzy matching to find the best match from the provided mapping.
    Args:
        education_input (str): The input string to match against the mapping.
        mapping (list): A li`st of Row objects containing level_references and level_scale.
    Returns:
        str: The best matching value from the mapping, or None if no match is found.
    """
    if education_input is None or len(education_input) == 0:
        return None
    education_input = _norm(", ".join(education_input)) if isinstance(education_input, list) else _norm(education_input)
    best_score = 0
    best_match = None
    for row in mapping:
        match_result = process.extractOne(query=education_input, choices=row.group_references, scorer=fuzz.token_set_ratio)
        if match_result and match_result[1] > best_score:
            best_score, best_match = match_result[1], row.group_name
    return best_match if best_score >= threshold else "Others"
